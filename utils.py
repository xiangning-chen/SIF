import numpy as np
import os
import os.path
import sys
import shutil
import torch
import torch.nn as nn
import torch.utils
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle

from models_binary import PRIMITIVES_BINARY, Network_Search
from models_triple import PRIMITIVES_TRIPLE, Network_Search_Triple


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def sample_arch():
	arch = {}
	arch['mlp'] = {}
	arch['mlp']['p'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['mlp']['q'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['binary'] = PRIMITIVES_BINARY[np.random.randint(len(PRIMITIVES_BINARY))]
	return arch


def sample_arch_triple():
	arch = {}
	arch['mlp'] = {}
	arch['mlp']['p'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['mlp']['q'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['mlp']['r'] = nn.Sequential(
		nn.Linear(1, 8),
		nn.Tanh(),
		nn.Linear(8, 1)).cuda()
	arch['triple'] = PRIMITIVES_TRIPLE[np.random.randint(len(PRIMITIVES_TRIPLE))]
	return arch


def update_arch(arch, cfg):
	flag = 0
	for p in arch.parameters():
		num = p.view(-1).size(0)
		p.data.add_(-p).add_(torch.tensor(cfg[flag:flag+num]).float().cuda().view(p.size()))
		flag += num


def load_arch(num_users, num_items, args):
	arch = {}
	arch['mlp'] = {}
	with open(os.path.join('experiments', args.dataset, args.arch, 'log.txt'), 'r') as f:
		for i, line in enumerate(f.readlines()):
			line = line.split()
			if 'genotype:' in line:
				arch['binary'] = line[-1]

	model = Network_Search(num_users, num_items, args.embedding_dim, args.weight_decay)
	model.load_state_dict(torch.load(os.path.join('experiments', args.dataset, args.arch, 'model.pt')))
	arch['mlp']['p'] = model.mlp_p
	arch['mlp']['q'] = model.mlp_q
	return arch


def load_arch_triple(num_ps, num_qs, num_rs, args):
	arch = {}
	arch['mlp'] = {}
	with open(os.path.join('experiments', args.dataset, args.arch, 'log.txt'), 'r') as f:
		for i, line in enumerate(f.readlines()):
			line = line.split()
			if 'genotype:' in line:
				arch['triple'] = line[-1]

	model = Network_Search_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay)
	model.load_state_dict(torch.load(os.path.join('experiments', args.dataset, args.arch, 'model.pt')))
	arch['mlp']['p'] = model.mlp_p
	arch['mlp']['q'] = model.mlp_q
	arch['mlp']['r'] = model.mlp_r
	return arch


def get_data_queue(args):
	users, items, labels = [], [], []
	if args.dataset == 'ml-100k':
		data_path = os.path.join(args.data, 'ml-100k', 'u.data')
	elif args.dataset == 'ml-1m':
		data_path = os.path.join(args.data, 'ml-1m', 'ratings.dat')
	elif args.dataset == 'ml-10m':
		data_path = os.path.join(args.data, 'ml-10m', 'ratings.dat')
	elif args.dataset == 'youtube-small':
		data_path = os.path.join(args.data, 'youtube-weighted-small.npy')

	if 'ml' in args.dataset:
        # movielens dataset
		with open(data_path, 'r') as f:
			for i, line in enumerate(f.readlines()):
				if args.dataset == 'ml-100k':
					line = line.split()
				elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
					line = line.split('::')
				users.append(int(line[0]) - 1)
				items.append(int(line[1]) - 1)
				labels.append(float(line[2]))
		labels = StandardScaler().fit_transform(np.reshape(labels, [-1,1])).flatten().tolist()

		print('user', max(users), min(users))
		print('item', max(items), min(items))

		users, items, labels = shuffle(users, items, labels)
		indices = list(range(len(users)))
		num_train = int(len(users) * args.train_portion)
		num_valid = int(len(users) * args.valid_portion)

		if not args.mode == 'libfm':
			data_queue = torch.utils.data.TensorDataset(torch.tensor(users), 
				torch.tensor(items), torch.tensor(labels))

			train_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size, 
				sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:num_train]), pin_memory=True)
			
			valid_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size, 
				sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[num_train:num_train+num_valid]), pin_memory=True)

			test_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size, 
				sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[num_train+num_valid:]), pin_memory=True)      

		else:
            # prepare data format for libfm
			data_queue = []
			for i in range(len(users)):
				data_queue.append({'user': str(users[i]), 'item': str(items[i])})
			
			v = DictVectorizer()
			data_queue = v.fit_transform(data_queue)
			train_queue = [data_queue[:num_train], np.array(labels[:num_train])]
			valid_queue = [data_queue[num_train:num_train+num_valid], np.array(labels[num_train:num_train+num_valid])]
			test_queue = [data_queue[num_train+num_valid:], np.array(labels[num_train+num_valid:])]

	else:
		# 3-d dataset
		[ps, qs, rs, labels] = np.load(data_path).tolist()
		labels = StandardScaler().fit_transform(np.reshape(labels, [-1,1])).flatten().tolist()
		
		ps = [int(i) for i in ps]
		qs = [int(i) for i in qs]
		rs = [int(i) for i in rs]
		print('p', max(ps), min(ps))
		print('q', max(qs), min(qs))
		print('r', max(rs), min(rs))
		
		ps, qs, rs, labels = shuffle(ps, qs, rs, labels)
		indices = list(range(len(ps)))
		num_train = int(len(ps) * args.train_portion)
		num_valid = int(len(ps) * args.valid_portion)

		if not args.mode == 'libfm':
			data_queue = torch.utils.data.TensorDataset(torch.tensor(ps), torch.tensor(qs), 
				torch.tensor(rs), torch.tensor(labels))

			train_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size, 
				sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:num_train]), pin_memory=True)
			
			valid_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size, 
				sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[num_train:num_train+num_valid]), pin_memory=True)

			test_queue = torch.utils.data.DataLoader(data_queue, batch_size=args.batch_size, 
				sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[num_train+num_valid:]), pin_memory=True)      

		else:
			# prepare data format for libfm
			data_queue = []
			for i in range(len(ps)):
				data_queue.append({'p': str(ps[i]), 'q': str(qs[i]), 'r': str(rs[i])})

			v = DictVectorizer()
			data_queue = v.fit_transform(data_queue)
			train_queue = [data_queue[:num_train], np.array(labels[:num_train])]
			valid_queue = [data_queue[num_train:num_train+num_valid], np.array(labels[num_train:num_train+num_valid])]
			test_queue = [data_queue[num_train+num_valid:], np.array(labels[num_train+num_valid:])]
		
	return train_queue, valid_queue, test_queue


