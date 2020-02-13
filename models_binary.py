import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math


PRIMITIVES_BINARY = ['plus', 'multiply', 'max', 'min', 'concat']
PRIMITIVES_NAS = [0, 2, 4, 8, 16]
SPACE_NAS = pow(len(PRIMITIVES_NAS), 5)
OPS = {
	'plus': lambda p, q: p + q,
	'multiply': lambda p, q: p * q,
	'max': lambda p, q: torch.max(torch.stack((p, q)), dim=0)[0],
	'min': lambda p, q: torch.min(torch.stack((p, q)), dim=0)[0],
	'concat': lambda p, q: torch.cat([p, q], dim=-1),
	'norm_0': lambda p: torch.ones_like(p),
	'norm_0.5': lambda p: torch.sqrt(torch.abs(p) + 1e-7),
	'norm_1': lambda p: torch.abs(p),
	'norm_2': lambda p: p ** 2,
	'I': lambda p: torch.ones_like(p),
	'-I': lambda p: -torch.ones_like(p),
	'sign': lambda p: torch.sign(p),
}


def constrain(p):
	c = torch.norm(p, p=2, dim=1, keepdim=True)
	c[c < 1] = 1.0
	p.data.div_(c)


def MixedBinary(embedding_p, embedding_q, weights, FC):
	return torch.sum(torch.stack([w * fc(OPS[primitive](embedding_p, embedding_q)) \
		for w,primitive,fc in zip(weights,PRIMITIVES_BINARY,FC)]), 0)


def _concat(xs):
	return torch.cat([x.view(-1) for x in xs])


class Virtue(nn.Module):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Virtue, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_dim = embedding_dim
		self.reg = reg
		self._UsersEmbedding = nn.Embedding(num_users, embedding_dim)
		self._ItemsEmbedding = nn.Embedding(num_items, embedding_dim)

	def compute_loss(self, inferences, labels, regs):
		labels = torch.reshape(labels, [-1,1])
		loss = F.mse_loss(inferences, labels)
		return loss + regs


class NCF(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(NCF, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)
		self._W = nn.Linear(2*embedding_dim, embedding_dim)
		# self._FC = nn.Sequential(
		# 	nn.Linear(embedding_dim, embedding_dim),
		# 	nn.Tanh(),
		# 	nn.Linear(embedding_dim, 1, bias=False))

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		constrain(next(self._W.parameters()))
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		gmf_out = users_embedding * items_embedding
		mlp_out = self._W(torch.cat([users_embedding, items_embedding], dim=-1))
		inferences = self._FC(F.tanh(gmf_out + mlp_out))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class DeepWide(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(DeepWide, self).__init__(num_users, num_items, embedding_dim, reg)
		# self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
		self._FC = nn.Sequential(
			nn.Linear(2*embedding_dim, embedding_dim),
			nn.ReLU(),
			nn.Linear(embedding_dim, 1, bias=False))

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(torch.cat([users_embedding, items_embedding], dim=-1))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class AltGrad(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(AltGrad, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(users_embedding * items_embedding)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class ConvNCF(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(ConvNCF, self).__init__(num_users, num_items, embedding_dim, reg)
		self.num_conv = int(math.log(embedding_dim, 2))
		self._Conv = []
		for i in range(self.num_conv-1):
			self._Conv.append(nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1))
			self._Conv.append(nn.ReLU())
		self._Conv.append(nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1))
		self._Conv = nn.Sequential(*self._Conv)

	def forward(self, users, items):
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		outer_result = torch.bmm(users_embedding.view(-1,self.embedding_dim,1), 
			items_embedding.view(-1,1,self.embedding_dim))

		outer_result = torch.unsqueeze(outer_result, 1)

		inferences = self._Conv(outer_result).view(-1, 1)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class Plus(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Plus, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(users_embedding + items_embedding)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class Max(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Max, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(OPS['max'](users_embedding, items_embedding))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class Min(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Min, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(OPS['min'](users_embedding, items_embedding))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class Conv(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Conv, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = []
		for i in range(self.embedding_dim):
			tmp = torch.zeros(users_embedding.size(0), 1, dtype=torch.float, device='cuda', requires_grad=False)
			for j in range(i+1):
				tmp += torch.reshape(users_embedding[:,j]*items_embedding[:,i-j], [-1,1])
			inferences.append(tmp)
		inferences = torch.cat(inferences, -1)
		inferences = self._FC(inferences)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class Outer(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(Outer, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim**2, 1, bias=False)

	def forward(self, users, items):
		constrain(next(self._FC.parameters()))
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = torch.bmm(users_embedding.view(-1,self.embedding_dim,1), 
			items_embedding.view(-1,1,self.embedding_dim)).view(-1,self.embedding_dim**2)
		inferences = self._FC(inferences)
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class AutoNeural(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, reg):
		super(AutoNeural, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = nn.Sequential(
			nn.Linear(2*embedding_dim, 2*embedding_dim),
			nn.Sigmoid(),
			nn.Linear(2*embedding_dim, 1))

	def forward(self, users, items):
		for p in self._FC.parameters():
			if len(p.size()) == 1: continue
			constrain(p)

		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(torch.cat([users_embedding,items_embedding], dim=-1))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))

		return inferences, regs

	def embedding_parameters(self):
		return list(self._UsersEmbedding.parameters()) + list(self._ItemsEmbedding.parameters())

	def mlp_parameters(self):
		return self._FC.parameters()


class NAS(Virtue):

	def __init__(self, num_users, num_items, embedding_dim, arch, reg):
		super(NAS, self).__init__(num_users, num_items, embedding_dim, reg)
		self._FC = []

		for i in range(len(arch)):
			if i == 0:
				self._FC.append(nn.Linear(2*embedding_dim, int(arch[i])))
			else:
				self._FC.append(nn.Linear(int(arch[i-1]), int(arch[i])))
			self._FC.append(nn.ReLU())
		if len(self._FC) == 0:
			self._FC.append(nn.Linear(2*embedding_dim, 1, bias=False))
		else:
			self._FC.append(nn.Linear(arch[-1], 1, bias=False))
		self._FC = nn.Sequential(*self._FC)

	def forward(self, users, items):
		users_embedding = self._UsersEmbedding(users)
		items_embedding = self._ItemsEmbedding(items)

		inferences = self._FC(torch.cat([users_embedding, items_embedding], dim=-1))
		regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
		return inferences, regs


class Network(Virtue):
    
    def __init__(self, num_users, num_items, embedding_dim, arch, reg):
        super(Network, self).__init__(num_users, num_items, embedding_dim, reg)
        self.arch = arch
        self.mlp_p = arch['mlp']['p']
        self.mlp_q = arch['mlp']['q']
        
        if arch['binary'] == 'concat':
            self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
        else:
            self._FC = nn.Linear(embedding_dim, 1, bias=False)
    
    def parameters(self):
        return list(self._UsersEmbedding.parameters()) + list(self._ItemsEmbedding.parameters()) + \
            list(self._FC.parameters())
    
    def forward(self, users, items):
        constrain(next(self._FC.parameters()))
        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)
        
        users_embedding_trans = self.mlp_p(users_embedding.view(-1,1)).view(users_embedding.size())
        items_embedding_trans = self.mlp_q(items_embedding.view(-1,1)).view(items_embedding.size())
        
        inferences = self._FC(OPS[self.arch['binary']](users_embedding_trans, items_embedding_trans))
        regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
        return inferences, regs


class Network_Search(Virtue):
    
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(Network_Search, self).__init__(num_users, num_items, embedding_dim, reg)
        self._FC = nn.ModuleList()
        for primitive in PRIMITIVES_BINARY:
            if primitive == 'concat':
                self._FC.append(nn.Linear(2*embedding_dim, 1, bias=False))
            else:
                self._FC.append(nn.Linear(embedding_dim, 1, bias=False))
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        self.mlp_p = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()
        self.mlp_q = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
            nn.Linear(8, 1)).cuda()
        self._arch_parameters = {}
        self._arch_parameters['mlp'] = {}
        self._arch_parameters['mlp']['p'] = self.mlp_p
        self._arch_parameters['mlp']['q'] = self.mlp_q
        self._arch_parameters['binary'] = Variable(torch.ones(len(PRIMITIVES_BINARY), 
            dtype=torch.float, device='cuda') / 2, requires_grad=True)
        self._arch_parameters['binary'].data.add_(
            torch.randn_like(self._arch_parameters['binary'])*1e-3)
    
    def arch_parameters(self):
        return list(self._arch_parameters['mlp']['p'].parameters()) + \
            list(self._arch_parameters['mlp']['q'].parameters()) + [self._arch_parameters['binary']]
    
    def new(self):
        model_new = Network_Search(self.num_users, self.num_items, self.embedding_dim, self.reg).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data = y.data.clone()
        return model_new
    
    def clip(self):
        m = nn.Hardtanh(0, 1)
        self._arch_parameters['binary'].data = m(self._arch_parameters['binary'])
    
    def binarize(self):
        self._cache = self._arch_parameters['binary'].clone()
        max_index = self._arch_parameters['binary'].argmax().item()
        for i in range(self._arch_parameters['binary'].size(0)):
            if i == max_index:
                self._arch_parameters['binary'].data[i] = 1.0
            else:
                self._arch_parameters['binary'].data[i] = 0.0
    
    def recover(self):
        self._arch_parameters['binary'].data = self._cache
        del self._cache

    def forward(self, users, items):
        for i in range(len(PRIMITIVES_BINARY)):
            constrain(next(self._FC[i].parameters()))

        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        users_embedding_trans = self._arch_parameters['mlp']['p'](users_embedding.view(-1,1)).view(users_embedding.size())
        items_embedding_trans = self._arch_parameters['mlp']['q'](items_embedding.view(-1,1)).view(items_embedding.size())

        # the weight is already binarized
        assert self._arch_parameters['binary'].sum() == 1.
        inferences = MixedBinary(users_embedding_trans, items_embedding_trans,
                                 self._arch_parameters['binary'], self._FC)

        regs = self.reg * (torch.norm(users_embedding) + torch.norm(items_embedding))
        return inferences, regs

    def genotype(self):
        genotype = PRIMITIVES_BINARY[self._arch_parameters['binary'].argmax().cpu().numpy()]
        genotype_p = F.softmax(self._arch_parameters['binary'], dim=-1)
        return genotype, genotype_p.cpu().detach()

    def step(self, users_train, items_train, labels_train, users_valid, 
		items_valid, labels_valid, lr, arch_optimizer, unrolled):
        self.zero_grad()
        arch_optimizer.zero_grad()

        # binarize before forward propagation
        self.binarize()
        if unrolled:
            loss = self._backward_step_unrolled(users_train, items_train, labels_train,
				users_valid, items_valid, labels_valid, lr)
        else:
            loss = self._backward_step(users_valid, items_valid, labels_valid)
        # restore weight before updating
        self.recover()
        arch_optimizer.step()
        return loss
    
    def _backward_step(self, users_valid, items_valid, labels_valid):
        inferences, regs = self(users_valid, items_valid)
        loss = self.compute_loss(inferences, labels_valid, regs)
        loss.backward()
        return loss
    
    def _backward_step_unrolled(self, users_train, items_train, labels_train,
		users_valid, items_valid, labels_valid, lr):
        unrolled_model = self._compute_unrolled_model(
			users_train, items_train, labels_train, lr)
        unrolled_inference, unrolled_regs = unrolled_model(users_valid, items_valid)
        unrolled_loss = unrolled_model.compute_loss(unrolled_inference, labels_valid, unrolled_regs)
        
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, users_train, items_train, labels_train)
        
        for g,ig in zip(dalpha,implicit_grads):
            g.sub_(lr, ig)
        
        for v,g in zip(self.arch_parameters(), dalpha):
            v.grad = g.clone()
        return unrolled_loss
    
    def _compute_unrolled_model(self, users_train, items_train, labels_train, lr):
        inferences, regs = self(users_train, items_train)
        loss = self.compute_loss(inferences, labels_train, regs)
        theta = _concat(self.parameters())
        dtheta = _concat(torch.autograd.grad(loss, self.parameters())) + self.reg * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(lr, dtheta))
        return unrolled_model
    
    def _construct_model_from_theta(self, theta):
        model_new = self.new()
        model_dict = self.state_dict()
        params, offset = {}, 0
        for k,v in self.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()
    
    def _hessian_vector_product(self, vector, users, items, labels, r=1e-2):
        R = r / _concat(vector).norm()
        for p,v in zip(self.parameters(), vector):
            p.data.add_(R, v)
        inferences, regs = self(users, items)
        loss = self.compute_loss(inferences, labels, regs)
        grads_p = torch.autograd.grad(loss, self.arch_parameters())

        for p,v in zip(self.parameters(), vector):
            p.data.sub_(2*R, v)
        inferences, regs = self(users, items)
        loss = self.compute_loss(inferences, labels, regs)
        grads_n = torch.autograd.grad(loss, self.arch_parameters())

        for p,v in zip(self.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x,y in zip(grads_p,grads_n)]






    






