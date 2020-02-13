import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from models_binary import OPS, constrain


PRIMITIVES_TRIPLE = ['0_plus_multiply', '0_plus_max', '0_plus_min', '0_plus_concat',
					 '0_multiply_plus', '0_multiply_max', '0_multiply_min', '0_multiply_concat',
					 '0_max_plus', '0_max_multiply', '0_max_min', '0_max_concat',
					 '0_min_plus', '0_min_multiply', '0_min_max', '0_min_concat',
					 '1_plus_multiply', '1_plus_max', '1_plus_min', '1_plus_concat',
					 '1_multiply_plus', '1_multiply_max', '1_multiply_min', '1_multiply_concat',
					 '1_max_plus', '1_max_multiply', '1_max_min', '1_max_concat',
					 '1_min_plus', '1_min_multiply', '1_min_max', '1_min_concat',
					 '2_plus_multiply', '2_plus_max', '2_plus_min', '2_plus_concat',
					 '2_multiply_plus', '2_multiply_max', '2_multiply_min', '2_multiply_concat',
					 '2_max_plus', '2_max_multiply', '2_max_min', '2_max_concat',
					 '2_min_plus', '2_min_multiply', '2_min_max', '2_min_concat',
					 'plus_plus', 'multiply_multiply', 'max_max', 'min_min', 'concat_concat',
					 ]
PRIMITIVES_NAS = [0, 2, 4, 8, 16]


def ops_triple(triple, p, q, r):
	if triple == 'plus_plus':
		return OPS['plus'](OPS['plus'](p, q), r)
	elif triple == 'multiply_multiply':
		return OPS['plus'](OPS['plus'](p, q), r)
	elif triple == 'max_max':
		return OPS['max'](OPS['max'](p, q), r)
	elif triple == 'min_min':
		return OPS['min'](OPS['min'](p, q), r)
	elif triple == 'concat_concat':
		return OPS['concat'](OPS['concat'](p, q), r)
	else:
		ops = triple.split('_')
		if ops[0] == '0':
			return OPS[ops[2]](OPS[ops[1]](p, q), r)
		elif ops[0] == '1':
			return OPS[ops[2]](OPS[ops[1]](p, r), q)
		elif ops[0] == '2':
			return OPS[ops[2]](OPS[ops[1]](q, r), p)


def _concat(xs):
	return torch.cat([x.view(-1) for x in xs])


def MixedTriple(embedding_p, embedding_q, embedding_r, weights, FC):
	return torch.sum(torch.stack([w * fc(ops_triple(primitive, embedding_p, embedding_q, embedding_r)) \
		for w,primitive,fc in zip(weights,PRIMITIVES_TRIPLE,FC)]), 0)


class Virtue_Triple(nn.Module):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(Virtue_Triple, self).__init__()
		self.num_ps = num_ps
		self.num_qs = num_qs
		self.num_rs = num_rs
		self.embedding_dim = embedding_dim
		self.reg = reg
		self._PsEmbedding = nn.Embedding(num_ps, embedding_dim)
		self._QsEmbedding = nn.Embedding(num_qs, embedding_dim)
		self._RsEmbedding = nn.Embedding(num_rs, embedding_dim)

	def compute_loss(self, inferences, labels, regs):
		labels = torch.reshape(labels, [-1,1])
		loss = F.mse_loss(inferences, labels)
		return loss + regs
	

class NCF_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(NCF_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)
		self._W = nn.Linear(3*embedding_dim, embedding_dim)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))
		constrain(next(self._W.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		gmf_out = ps_embedding * qs_embedding * rs_embedding
		mlp_out = self._W(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		inferences = self._FC(F.relu(gmf_out + mlp_out))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs


class DeepWide_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(DeepWide_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(3*embedding_dim, 1, bias=False)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs


class CP(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(CP, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(ps_embedding * qs_embedding * rs_embedding)
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs


class TuckER(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(TuckER, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		w = torch.empty(embedding_dim, embedding_dim, embedding_dim)
		nn.init.xavier_uniform_(w)
		self._W = torch.nn.Parameter(torch.tensor(w, dtype=torch.float, device='cuda', requires_grad=True))

	def forward(self, ps, qs, rs):
		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		W_after_p = torch.mm(ps_embedding, self._W.view(ps_embedding.size(1), -1))
		W_after_p = W_after_p.view(-1, rs_embedding.size(1), qs_embedding.size(1))
		W_after_r = torch.bmm(rs_embedding.view(-1,1,rs_embedding.size(1)), W_after_p)
		W_after_q = torch.bmm(W_after_r, qs_embedding.view(-1,qs_embedding.size(1),1))
		inferences = W_after_q.view(-1,1)
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs


class NAS_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, arch, reg):
		super(NAS_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = []

		for i in range(len(arch)):
			if i == 0:
				self._FC.append(nn.Linear(3*embedding_dim, int(arch[i])))
			else:
				self._FC.append(nn.Linear(int(arch[i-1]), int(arch[i])))
			self._FC.append(nn.ReLU())
		if len(self._FC) == 0:
			self._FC.append(nn.Linear(3*embedding_dim, 1, bias=False))
		else:
			self._FC.append(nn.Linear(arch[-1], 1, bias=False))
		self._FC = nn.Sequential(*self._FC)

	def forward(self, ps, qs, rs):
		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding, qs_embedding, rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs
	

class AutoNeural_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(AutoNeural_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.Sequential(
			nn.Linear(3*embedding_dim, 3*embedding_dim),
			nn.Sigmoid(),
			nn.Linear(3*embedding_dim, 1))

	def forward(self, ps, qs, rs):
		for p in self._FC.parameters():
			if len(p.size()) == 1: continue
			constrain(p)

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		inferences = self._FC(torch.cat([ps_embedding,qs_embedding,rs_embedding], dim=-1))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))

		return inferences, regs

	def embedding_parameters(self):
		return list(self._PsEmbedding.parameters()) + list(self._QsEmbedding.parameters()) + \
			list(self._RsEmbedding.parameters())

	def mlp_parameters(self):
		return self._FC.parameters()


class Network_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, arch, reg):
		super(Network_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self.arch = arch
		self.mlp_p = arch['mlp']['p']
		self.mlp_q = arch['mlp']['q']
		self.mlp_r = arch['mlp']['r']

		if arch['triple'] == 'concat_concat':
			self._FC = nn.Linear(3*embedding_dim, 1, bias=False)
		elif 'concat' in arch['triple']:
			self._FC = nn.Linear(2*embedding_dim, 1, bias=False)
		else:
			self._FC = nn.Linear(embedding_dim, 1, bias=False)

	def parameters(self):
		return list(self._PsEmbedding.parameters()) + list(self._QsEmbedding.parameters()) + \
			list(self._RsEmbedding.parameters()) + list(self._FC.parameters())

	def forward(self, ps, qs, rs):
		constrain(next(self._FC.parameters()))
		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		ps_embedding_trans = self.mlp_p(ps_embedding.view(-1,1)).view(ps_embedding.size())
		qs_embedding_trans = self.mlp_q(qs_embedding.view(-1,1)).view(qs_embedding.size())
		rs_embedding_trans = self.mlp_r(rs_embedding.view(-1,1)).view(rs_embedding.size())

		inferences = self._FC(ops_triple(self.arch['triple'], ps_embedding_trans, 
			qs_embedding_trans, rs_embedding_trans))
		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + \
			torch.norm(rs_embedding))
		return inferences, regs


class Network_Search_Triple(Virtue_Triple):

	def __init__(self, num_ps, num_qs, num_rs, embedding_dim, reg):
		super(Network_Search_Triple, self).__init__(num_ps, num_qs, num_rs, embedding_dim, reg)
		self._FC = nn.ModuleList()
		for primitive in PRIMITIVES_TRIPLE:
			if primitive == 'concat_concat':
				self._FC.append(nn.Linear(3*embedding_dim, 1, bias=False))
			elif 'concat' in primitive:
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
		self.mlp_r = nn.Sequential(
			nn.Linear(1, 8),
			nn.Tanh(),
			nn.Linear(8, 1)).cuda()
		self._arch_parameters = {}
		self._arch_parameters['mlp'] = {}
		self._arch_parameters['mlp']['p'] = self.mlp_p
		self._arch_parameters['mlp']['q'] = self.mlp_q
		self._arch_parameters['mlp']['r'] = self.mlp_r
		self._arch_parameters['triple'] = Variable(torch.ones(len(PRIMITIVES_TRIPLE), 
            dtype=torch.float, device='cuda') / 2, requires_grad=True)
		self._arch_parameters['triple'].data.add_(
            torch.randn_like(self._arch_parameters['triple'])*1e-3)

	def arch_parameters(self):
		return list(self._arch_parameters['mlp']['p'].parameters()) + \
			   list(self._arch_parameters['mlp']['q'].parameters()) + \
			   list(self._arch_parameters['mlp']['r'].parameters()) + \
			   [self._arch_parameters['triple']]

	def new(self):
		model_new = Network_Search_Triple(self.num_ps, self.num_qs, self.num_rs, self.embedding_dim, self.reg).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data = y.data.clone()
			try:
				x.grad = y.grad.clone()
			except:
				pass
		return model_new

	def clip(self):
		m = nn.Hardtanh(0, 1)
		self._arch_parameters['triple'].data = m(self._arch_parameters['triple'])
	
	def binarize(self):
		self._cache = self._arch_parameters['triple'].clone()
		max_index = self._arch_parameters['triple'].argmax().item()
		for i in range(self._arch_parameters['triple'].size(0)):
			if i == max_index:
				self._arch_parameters['triple'].data[i] = 1.0
			else:
				self._arch_parameters['triple'].data[i] = 0.0
	
	def recover(self):
		self._arch_parameters['triple'].data = self._cache
		del self._cache

	def forward(self, ps, qs, rs):
		for i in range(len(PRIMITIVES_TRIPLE)):
			constrain(next(self._FC[i].parameters()))

		ps_embedding = self._PsEmbedding(ps)
		qs_embedding = self._QsEmbedding(qs)
		rs_embedding = self._RsEmbedding(rs)

		ps_embedding_trans = self._arch_parameters['mlp']['p'](ps_embedding.view(-1,1)).view(ps_embedding.size())
		qs_embedding_trans = self._arch_parameters['mlp']['q'](qs_embedding.view(-1,1)).view(qs_embedding.size())
		rs_embedding_trans = self._arch_parameters['mlp']['r'](rs_embedding.view(-1,1)).view(rs_embedding.size())

		# the weight is already binarized
		assert self._arch_parameters['triple'].sum() == 1.
		inferences = MixedTriple(ps_embedding_trans, qs_embedding_trans, rs_embedding_trans,
			self._arch_parameters['triple'], self._FC)

		regs = self.reg * (torch.norm(ps_embedding) + torch.norm(qs_embedding) + torch.norm(rs_embedding))
		return inferences, regs

	def genotype(self):
		genotype = PRIMITIVES_TRIPLE[self._arch_parameters['triple'].argmax().cpu().numpy()]
		genotype_p = F.softmax(self._arch_parameters['triple'], dim=-1)
		return genotype, genotype_p.cpu().detach()
	
	def step(self, p_train, q_train, r_train, labels_train, p_valid, q_valid, 
		r_valid, labels_valid, lr, arch_optimizer, unrolled):
		self.zero_grad()
		arch_optimizer.zero_grad()

		# binarize before forward propagation
		self.binarize()
		if unrolled:
			loss = self._backward_step_unrolled(p_train, q_train, r_train, 
				labels_train, p_valid, q_valid, r_valid, labels_valid, lr)
		else:
			loss = self._backward_step(p_valid, q_valid, r_valid, labels_valid)
		# restore weight before updating
		self.recover()
		arch_optimizer.step()
		return loss

	def _backward_step(self, p_valid, q_valid, r_valid, labels_valid):
		inferences, regs = self(p_valid, q_valid, r_valid)
		loss = self.compute_loss(inferences, labels_valid, regs)
		loss.backward()
		return loss

	def _backward_step_unrolled(self, p_train, q_train, r_train, labels_train, 
		p_valid, q_valid, r_valid, labels_valid, lr):
		unrolled_model = self._compute_unrolled_model(
			p_train, q_train, r_train, labels_train, lr)
		unrolled_inference, unrolled_regs = unrolled_model(p_valid, q_valid, r_valid)
		unrolled_loss = unrolled_model.compute_loss(unrolled_inference, labels_valid, unrolled_regs)

		unrolled_loss.backward()
		dalpha = [v.grad for v in unrolled_model.arch_parameters()]
		vector = [v.grad for v in unrolled_model.parameters()]
		implicit_grads = self._hessian_vector_product(vector, p_train, q_train, r_train, labels_train)

		for g, ig in zip(dalpha,implicit_grads):
			g.sub_(lr, ig)

		for v, g in zip(self.arch_parameters(), dalpha):
			v.grad = g.clone()
		return unrolled_loss

	def _compute_unrolled_model(self, p_train, q_train, r_train, labels_train, lr):
		inferences, regs = self(p_train, q_train, r_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		theta = _concat(self.parameters())
		dtheta = _concat(torch.autograd.grad(loss, self.parameters())) + \
			self.reg * theta
		unrolled_model = self._construct_model_from_theta(
			theta.sub(lr, dtheta))
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

	def _hessian_vector_product(self, vector, p_train, q_train, r_train, labels_train, r=1e-2):
		R = r / _concat(vector).norm()
		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)
		inferences, regs = self(p_train, q_train, r_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		grads_p = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.sub_(2*R, v)
		inferences, regs = self(p_train, q_train, r_train)
		loss = self.compute_loss(inferences, labels_train, regs)
		grads_n = torch.autograd.grad(loss, self.arch_parameters())

		for p,v in zip(self.parameters(), vector):
			p.data.add_(R, v)

		return [(x-y).div_(2*R) for x,y in zip(grads_p,grads_n)]
