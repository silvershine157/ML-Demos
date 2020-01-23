import torch
from model import GaussianMLP, BasicMLP

def test1():
	x = torch.zeros([4, 1])
	gmlp = GaussianMLP()
	bmlp = BasicMLP()
	mu, var = gmlp(x)
	print(mu.shape)
	print(var.shape)
	mu = bmlp(x)
	print(mu.shape)

def test2():
	x = torch.linspace(0, 2, 100).view([100, 1])
	y = torch.abs(x-1.0)
	train_ensmeble(BasicMLP, x, y, 5)


def train_ensmeble(module, x, y, ens_size):
	nets = [module() for _ in range(ens_size)]
	for net_i, net in enumerate(nets):
		loss = train_single(net, x, y)
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


def train_single(net, x, y):
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
		optimizer.zero_grad()
		loss = net.loss(x, y)
		loss.backward()
		optimizer.step()
		#print("loss: %.3f"%(loss.item()))
	loss = loss.item() # final loss
	return loss

