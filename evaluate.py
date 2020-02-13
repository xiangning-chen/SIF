import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def evaluate(model, test_queue):
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        for users, items, labels in test_queue:
            inferences, _ = model(users.cuda(), items.cuda())
            xs.append(inferences.flatten())
            ys.append(labels.cuda())
        mse = F.mse_loss(torch.cat(xs), torch.cat(ys))
        rmse = torch.sqrt(mse)
    return rmse.cpu().detach().item()


def evaluate_triple(model, test_queue):
	model.eval()
	xs, ys = [], []
	with torch.no_grad():
		for ps, qs, rs, labels in test_queue:
			inferences, _ = model(ps.cuda(), qs.cuda(), rs.cuda())
			xs.append(inferences.flatten())
			ys.append(labels.cuda())
		mse = F.mse_loss(torch.cat(xs), torch.cat(ys))
		rmse = torch.sqrt(mse)
	return rmse.cpu().detach().item()


def evaluate_hr_ndcg(model, test_queue, topk=10):
	model.eval()
	with torch.no_grad():
		users, items, _ = test_queue
		users = users.cpu().tolist()
		hrs, ndcgs = [], []
		
		inferences_dict = {}
		
		users_all, items_all = [], []
		for user in list(set(users)):
			users_all += [user] * model.num_items
			items_all += list(range(model.num_items))
		inferences, _ = model(torch.tensor(users_all).cuda(), torch.tensor(items_all).cuda())
		inferences = inferences.detach().cpu().tolist()
		for i, user in enumerate(list(set(users))):
			inferences_dict[user] = inferences[i*model.num_items:(i+1)*model.num_items]

		for i, user in enumerate(users):
			inferences = inferences_dict[user]
			score = inferences[items[i]]
			rank = 0
			for s in inferences:
				if score < s:
					rank += 1
			if rank < topk:
				hr = 1.0
				ndcg = math.log(2) / math.log(rank+2)
			else:
				hr = 0.0
				ndcg = 0.0
			hrs.append(hr)
			ndcgs.append(ndcg)
	return np.mean(hrs), np.mean(ndcgs)
