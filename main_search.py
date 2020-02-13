import os
import sys
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F
from torch.autograd import Variable
import time
import utils

from train import get_arch_performance, train_search
from train import get_arch_performance_triple, train_search_triple
from models_binary import PRIMITIVES_BINARY, Network_Search
from models_triple import PRIMITIVES_TRIPLE, Network_Search_Triple
from evaluate import evaluate, evaluate_triple

from itertools import chain
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser(description="Search.")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--lr', type=float, default=5e-2, help='init learning rate')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--opt', type=str, default='Adagrad', help='choice of opt')
parser.add_argument('--batch_size', type=int, default=512, help='choose batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--search_epochs', type=int, default=100, help='num of searching epochs')
parser.add_argument('--train_epochs', type=int, default=10000, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.25, help='portion of validation data')
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
parser.add_argument('--mode', type=str, default='sif', help='choose how to search')
parser.add_argument('--embedding_dim', type=int, default=2, help='dimension of embedding')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
args = parser.parse_args()


save_name = 'experiments/{}/search-{}-{}-{}-{}-{}-{}-{}-{}'.format(args.dataset, time.strftime("%Y%m%d-%H%M%S"),
	args.mode, args.save, args.embedding_dim, args.opt, args.lr, args.arch_lr, args.seed)
if args.unrolled:
    save_name += '-unrolled'
save_name += '-' + str(np.random.randint(10000))
utils.create_exp_dir(save_name, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
	torch.set_default_tensor_type(torch.FloatTensor)
	torch.set_num_threads(3)
	if not torch.cuda.is_available():
		logging.info('no gpu device available')
		sys.exit(1)
	
	np.random.seed(args.seed)
	torch.cuda.set_device(args.gpu)
	cudnn.benchmark = True
	torch.manual_seed(args.seed)
	cudnn.enabled = True
	torch.cuda.manual_seed(args.seed)
	logging.info('gpu device = %d' % args.gpu)
	logging.info("args = %s", args)
	
	data_start = time.time()
	if args.dataset == 'ml-100k':
		num_users = 943
		num_items = 1682
		dim = 2
	elif args.dataset == 'ml-1m':
		num_users = 6040
		num_items = 3952
		dim = 2
	elif args.dataset == 'ml-10m':
		num_users = 71567
		num_items = 65133
		dim = 2
	elif args.dataset == 'youtube-small':
		num_ps = 600
		num_qs = 14340
		num_rs = 5
		dim = 3
    
	train_queue, valid_queue, test_queue = utils.get_data_queue(args)
	logging.info('prepare data finish! [%f]' % (time.time()-data_start))


	if args.mode == 'random':
		search_start = time.time()
		best_arch, best_rmse = None, 100000
		
		archs = []
		for search_epoch in range(args.search_epochs):
			arch = utils.sample_arch() if dim == 2 else utils.sample_arch_triple()
			archs.append(arch)

		for search_epoch in range(args.search_epochs):	
			arch = archs[search_epoch]
			arch_start = time.time()

			arch['triple'] = '2_multiply_concat'

			if dim == 2:
				rmse = get_arch_performance(arch, num_users, num_items, train_queue, test_queue, args, True)
			elif dim == 3:
				rmse = get_arch_performance_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args, True)
			if rmse < best_rmse:
				best_arch, best_rmse = arch, rmse

			op = arch['binary'] if dim == 2 else arch['triple']
			best_op = best_arch['binary'] if dim == 2 else best_arch['triple']

			logging.info('search_epoch: %d finish, arch: %s, rmse: %.4f, arch spent: %.4f' % (
				search_epoch, op, rmse, time.time()-arch_start))
			logging.info('search_epoch: %d, best_arch: %s, best_rmse: %.4f, time spent: %.4f' % (
				search_epoch, best_op, best_rmse, time.time()-search_start))
	

	elif args.mode == 'hyperopt':
		start = time.time()
		from hyperopt import fmin, tpe, hp
		def get_cfg_performance(cfg):
			arch = {}; arch['mlp'] = {}
			if dim == 2:
				arch = utils.sample_arch()
				arch['binary'] = cfg['binary']
				utils.update_arch(arch['mlp']['p'], cfg['mlp_p'])
				utils.update_arch(arch['mlp']['q'], cfg['mlp_q'])
				rmse = get_arch_performance(arch, num_users, num_items, train_queue, test_queue, args, True)
			elif dim == 3:
				arch = utils.sample_arch_triple()
				arch['triple'] = cfg['triple']
				utils.update_arch(arch['mlp']['p'], cfg['mlp_p'])
				utils.update_arch(arch['mlp']['q'], cfg['mlp_q'])
				utils.update_arch(arch['mlp']['r'], cfg['mlp_r'])
				rmse = get_arch_performance_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args, True)

			op = arch['binary'] if dim == 2 else arch['triple']
			logging.info('arch: %s, rmse: %.4f, time spent: %.4f' % (arch, rmse, time.time()-start))
			return rmse

		if dim == 2:
			space = {'mlp_p': [hp.uniform('mlp_p%d'%i, -1.0, 1.0) for i in range(25)],
					 'mlp_q': [hp.uniform('mlp_q%d'%i, -1.0, 1.0) for i in range(25)],
					 'binary': hp.choice('binary', PRIMITIVES_BINARY),
					 }
		elif dim == 3:
			space = {'mlp_p': [hp.uniform('mlp_p%d'%i, -1.0, 1.0) for i in range(25)],
					 'mlp_q': [hp.uniform('mlp_q%d'%i, -1.0, 1.0) for i in range(25)],
					 'mlp_r': [hp.uniform('mlp_r%d'%i, -1.0, 1.0) for i in range(25)],
					 'triple': hp.choice('triple', PRIMITIVES_TRIPLE),
					 }
		best = fmin(fn=get_cfg_performance,
					space=space,
					algo=tpe.suggest,
					max_evals=200)


	elif 'sif' in args.mode:
		search_start = time.time()
		if dim == 2:
			model = Network_Search(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif dim == 3:
			model = Network_Search_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		
		g, gp = model.genotype()
		logging.info('genotype: %s' % g)
		logging.info('genotype_p: %s' % gp)

		optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
		arch_optimizer = torch.optim.Adam(model.arch_parameters(), args.arch_lr)

		for search_epoch in range(args.search_epochs):
			if dim == 2:
				g, gp, loss = train_search(train_queue, valid_queue, model, optimizer, arch_optimizer, args)
				model.binarize()
				rmse = evaluate(model, test_queue)
				model.recover()
			elif dim == 3:
				g, gp, loss = train_search_triple(train_queue, valid_queue, model, optimizer, arch_optimizer, args)
				model.binarize()
				rmse = evaluate_triple(model, test_queue)
				model.recover()

			logging.info('search_epoch: %d, loss: %.4f, rmse: %.4f, time spent: %.4f' % (
				search_epoch, loss, rmse, time.time()-search_start))

			logging.info('genotype: %s' % g)  
			logging.info('genotype_p: %s' % gp)
			torch.save(model.state_dict(), os.path.join(save_name, 'model.pt'))


if __name__ == '__main__':
    main()
	





















