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

from train import train, train_triple, get_arch_performance_triple
from models_binary import NCF, DeepWide, AltGrad, ConvNCF, Plus, Max, Min, Conv, Outer, Network
from models_triple import NCF_Triple, DeepWide_Triple, CP, TuckER, Network_Triple
from evaluate import evaluate, evaluate_triple

from itertools import chain
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser(description="Run.")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--lr', type=float, default=0.05, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--opt', type=str, default='Adagrad', help='choice of opt')
parser.add_argument('--batch_size', type=int, default=512, help='choose batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--train_epochs', type=int, default=10000, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.25, help='portion of validation data')
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
parser.add_argument('--mode', type=str, default='deepwide', help='search or single mode')
parser.add_argument('--embedding_dim', type=int, default=2, help='dimension of embedding')
parser.add_argument('--arch', type=str, default='search-20200212-141611-sif-EXP-2-Adagrad-0.05-0.0003-1-8107',
                    help='choose which arch to evaluate')
args = parser.parse_args()


save_name = 'experiments/{}/evaluate-{}-{}-{}-{}-{}-{}-{}'.format(args.dataset, time.strftime("%Y%m%d-%H%M%S"),
	args.mode, args.save, args.embedding_dim, args.opt, args.lr, args.seed)
if args.mode == 'sif':
    # evaluate searched architecture
    save_name += '-arch-"' + args.arch + '"'
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
	
	if args.mode == 'libfm':
		start = time.time()
		from tffm import TFFMRegressor
		import tensorflow as tf
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
		model = TFFMRegressor(order=dim,
								rank=args.embedding_dim,
								optimizer=tf.train.AdagradOptimizer(learning_rate=args.lr),
								n_epochs=args.train_epochs,
								batch_size=args.batch_size,
								init_std=0.001,
								reg=args.weight_decay,
								input_type='sparse',
								log_dir=os.path.join(save_name, 'libfm-log'))
		model.fit(train_queue[0], train_queue[1], show_progress=True)
		inferences = model.predict(test_queue[0])
		mse = mean_squared_error(test_queue[1], inferences)
		rmse = np.sqrt(mse)
		logging.info('rmse: %.4f[%.4f]' % (rmse, time.time()-start))
    
	else:
		start = time.time()
		if args.mode == 'ncf':
			if dim == 2:
				model = NCF(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
			elif dim == 3:
				model = NCF_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'deepwide':
			if dim == 2:
				model = DeepWide(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
			elif dim == 3:
				model = DeepWide_Triple(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'altgrad':
			model = AltGrad(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'convncf':
			model = ConvNCF(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'outer':
			model = Outer(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'conv':
			model = Conv(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'plus':
			model = Plus(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'max':
			model = Max(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'min':
			model = Min(num_users, num_items, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'cp':
			model = CP(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'tucker':
			model = TuckER(num_ps, num_qs, num_rs, args.embedding_dim, args.weight_decay).cuda()
		elif args.mode == 'sif':
			if dim == 2:
				arch = utils.load_arch(num_users, num_items, args)
				print(next(arch['mlp']['p'].parameters()))
				model = Network(num_users, num_items, args.embedding_dim, arch, args.weight_decay).cuda()
			elif dim == 3:
				arch = utils.load_arch_triple(num_ps, num_qs, num_rs, args)
				model = Network_Triple(num_ps, num_qs, num_rs, args.embedding_dim, arch, args.weight_decay).cuda()
		logging.info('build model finish! [%f]' % (time.time()-start))

		optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
		if dim == 2:
			train(model, train_queue, test_queue, optimizer, args)
			rmse = evaluate(model, test_queue)
		elif dim == 3:
			train_triple(model, train_queue, test_queue, optimizer, args)
			rmse = evaluate_triple(model, test_queue)
		logging.info('rmse: %.4f' % rmse)

	
        
if __name__ == '__main__':
    main()
	
	





















