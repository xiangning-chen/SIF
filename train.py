import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models_binary import Network
from models_triple import Network_Triple

import logging
from time import time
import math
from evaluate import evaluate, evaluate_triple


def train_search(train_queue, valid_queue, model, optimizer, arch_optimizer, args):
	model.train()
	losses = []
	for step, (users_train, items_train, labels_train) in enumerate(train_queue):
		users_train = users_train.cuda()
		items_train = items_train.cuda()
		labels_train = labels_train.cuda()
		users_valid, items_valid, labels_valid = next(iter(valid_queue))
		users_valid = users_valid.cuda()
		items_valid = items_valid.cuda()
		labels_valid = labels_valid.cuda()

		if args.mode == 'sif-no-auto':
			loss_valid = model.step(users_train, items_train, labels_train, users_train,
				items_train, labels_train, args.lr, arch_optimizer, args.unrolled)
		else:
			loss_valid = model.step(users_train, items_train, labels_train, users_valid,
				items_valid, labels_valid, args.lr, arch_optimizer, args.unrolled)
		optimizer.zero_grad()
		arch_optimizer.zero_grad()

		model.binarize()
		inferences, regs = model(users_train, items_train)
		loss = model.compute_loss(inferences, labels_train, regs)
		loss.backward()
		model.recover()
		nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		optimizer.step()
		# restrict to [0,1]
		model.clip()
		optimizer.zero_grad()
		arch_optimizer.zero_grad()
		losses.append(loss.cpu().detach().item())

	g, gp = model.genotype()
	return g, gp, np.mean(losses)


def train_search_triple(train_queue, valid_queue, model, optimizer, arch_optimizer, args):
	model.train()
	losses = []
	for step, (ps_train, qs_train, rs_train, labels_train) in enumerate(train_queue):
		ps_train = ps_train.cuda()
		qs_train = qs_train.cuda()
		rs_train = rs_train.cuda()
		labels_train = labels_train.cuda()
		ps_valid, qs_valid, rs_valid, labels_valid = next(iter(valid_queue))
		ps_valid = ps_valid.cuda()
		qs_valid = qs_valid.cuda()
		rs_valid = rs_valid.cuda()
		labels_valid = labels_valid.cuda()

		if args.mode == 'sif-no-auto':
			loss_valid = model.step(ps_train, qs_train, rs_train, labels_train, ps_train,
				qs_train, rs_train, labels_train, args.lr, arch_optimizer, args.unrolled)
		else:
			loss_valid = model.step(ps_train, qs_train, rs_train, labels_train, ps_valid,
				qs_valid, rs_valid, labels_valid, args.lr, arch_optimizer, args.unrolled)
		optimizer.zero_grad()
		arch_optimizer.zero_grad()

		model.binarize()
		inferences, regs = model(ps_train, qs_train, rs_train)
		loss = model.compute_loss(inferences, labels_train, regs)
		loss.backward()
		model.recover()
		nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		optimizer.step()
		# restrict to [0,1]
		model.clip()
		optimizer.zero_grad()
		arch_optimizer.zero_grad()
		losses.append(loss.cpu().detach().item())

	g, gp = model.genotype()
	return g, gp, np.mean(losses)


def train(model, train_queue, test_queue, optimizer, args, show=True):
	losses = []
	start = time()
	model.train()
	for train_epoch in range(args.train_epochs):
		temp = []
		for (users_train, items_train, labels_train) in train_queue:
			inferences, regs = model(users_train.cuda(), items_train.cuda())
			loss = model.compute_loss(inferences, labels_train.cuda(), regs)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			optimizer.step()
			optimizer.zero_grad()
			model.zero_grad()
			temp.append(loss.cpu().detach().item())
		losses.append(np.mean(temp))

		if show:
			logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
				train_epoch, losses[-1], evaluate(model, test_queue), time()-start))

		if train_epoch > 100:
			if (losses[-2]-losses[-1])/losses[-1] < 1e-4/(len(train_queue)*args.batch_size) or np.isnan(losses[-1]):
				break


def train_triple(model, train_queue, test_queue, optimizer, args, show=True):
	losses = []
	start = time()
	model.train()
	for train_epoch in range(args.train_epochs):
		temp = []
		for (ps_train, qs_train, rs_train, labels_train) in train_queue:
			inferences, regs = model(ps_train.cuda(), qs_train.cuda(), rs_train.cuda())
			loss = model.compute_loss(inferences, labels_train.cuda(), regs)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			optimizer.step()
			optimizer.zero_grad()
			model.zero_grad()
			temp.append(loss.cpu().detach().item())
		losses.append(np.mean(temp))

		if show:
			logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
				train_epoch, losses[-1], evaluate_triple(model, test_queue), time()-start))

		if train_epoch > 100:
			if (losses[-2]-losses[-1])/losses[-1] < 1e-4/(len(train_queue)*args.batch_size) or np.isnan(losses[-1]):
				break


def get_arch_performance(arch, num_users, num_items, train_queue, test_queue, args, show=False):
	model = Network(num_users, num_items, args.embedding_dim, arch, args.weight_decay).cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
	train(model, train_queue, test_queue, optimizer, args, show)
	return evaluate(model, test_queue)


def get_arch_performance_triple(arch, num_ps, num_qs, num_rs, train_queue, test_queue, args, show=False):
	model = Network_Triple(num_ps, num_qs, num_rs, args.embedding_dim, arch, args.weight_decay).cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
	train_triple(model, train_queue, test_queue, optimizer, args, show)
	return evaluate_triple(model, test_queue)


























