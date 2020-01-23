import torch
from model import GaussianMLP, BasicMLP


def train_single(net, x, y, epsilon):
	'''
	net: GaussianMLP | BasicMLP
	x: [B, 1]
	y: [B, 1]
	---
	loss: float
	'''
	net.train()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
	n_iters = 1000
	for iter_i in range(n_iters):
		if epsilon is None:
			optimizer.zero_grad()
			loss = net.loss(x, y)
			loss.backward()
			optimizer.step()
		else:
			# adversarial training
			optimizer.zero_grad()
			x_temp = x.clone()
			x_temp.requires_grad_(True)
			orig_loss = net.loss(x_temp, y)
			orig_loss.backward()
			x_adv = x + epsilon*torch.sign(x_temp.grad)
			optimizer.zero_grad()
			adv_loss = net.loss(x_adv, y)
			loss = net.loss(x, y) + adv_loss
			loss.backward()
			optimizer.step()
	loss = loss.item() # final loss
	return loss


def train_ensmeble(module, x, y, ens_size, epsilon=None):
	nets = [module(d_hidden=100) for _ in range(ens_size)]
	for net_i, net in enumerate(nets):
		loss = train_single(net, x, y, epsilon)
		print('Net %d: final loss = %.3f'%(net_i, loss))
	return nets


def ensemble_prediction(nets, x):
	'''
	nets: list of BasicMLP | GaussianMLP
	x: [B, 1]
	---
	mu_ens: [B, 1]
	var_ens: [B, 1]
	'''
	ens_size = len(nets)
	with torch.no_grad():
		if isinstance(nets[0], BasicMLP):
			mu_all = []
			for net in nets:
				net.eval()
				mu = net(x)
				mu_all.append(mu)
			mu_all = torch.stack(mu_all, dim=0) # [ens_size, B, 1]
			mu_ens = torch.mean(mu_all, dim=0) # [B, 1]
			var_ens = torch.std(mu_all, dim=0)**2 # [B, 1]
		else:
			# GaussianMLP
			mu_all, var_all = [], []
			for net in nets:
				net.eval()
				mu, var = net(x)
				mu_all.append(mu)
				var_all.append(var)
			mu_all = torch.stack(mu_all, dim=0) # [ens_size, B, 1]
			var_all = torch.stack(var_all, dim=0) # [ens_size, B, 1]
			mu_ens = torch.mean(mu_all, dim=0) # [B, 1]
			var_ens = torch.mean(var_all + mu_all**2, dim=0) -  mu_ens**2 # [B, 1]
		mu_ens = mu_ens.numpy()
		var_ens = var_ens.numpy()

	return mu_ens, var_ens

