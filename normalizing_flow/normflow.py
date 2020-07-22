import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

'''
<convention>
f: x -> z
f_inv: z -> x
'''

use_gpu = True
device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

class Flow(nn.Module):
	def __init__(self):
		super(Flow, self).__init__()

	def f(self, x):
		raise NotImplementedError
		return z

	def f_inv(self, z):
		raise NotImplementedError
		return x

	def log_det_jac(self, x):
		raise NotImplementedError
		return 0.0

class IdentityFlow(Flow):
	# useless module! only for debugging.
	def __init__(self):
		super(IdentityFlow, self).__init__()

	def f(self, x):
		z = x
		return z

	def f_inv(self, z):
		x = z
		return x

	def log_det_jac(self, x):
		'''
		x: [B, ...]
		---
		res: [B]
		'''
		B = x.size(0)
		res = torch.zeros(B, device=device)
		return res

class CompositeFlow(Flow):
	def __init__(self, flow_list):
		super(CompositeFlow, self).__init__()
		self.subflows = nn.ModuleList(flow_list)

	def f(self, x):
		for i in range(len(self.subflows)):
			x = self.subflows[i].f(x)
		z = x
		return z

	def f_inv(self, z):
		for i in reversed(range(len(self.subflows))):
			z = self.subflows[i].f_inv(z)
		x = z
		return x

	def log_det_jac(self, x):
		'''
		x: [B, ...]
		---
		res: [B]
		'''
		B = x.size(0)
		res = torch.zeros(B, device=device)
		for i in range(len(self.subflows)):
			res += self.subflows[i].log_det_jac(x) # exploit chain rule
			x = self.subflows[i].f(x)
		return res

class AffineCoupling1D(Flow):
	def __init__(self, full_dim, change_first):
		super(AffineCoupling1D, self).__init__()
		self.change_first = change_first
		self.half_dim = full_dim//2
		hidden_dim = 1000
		self.net = nn.Sequential(
			nn.Linear(self.half_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),	
			nn.Linear(hidden_dim, full_dim)
		)
		self.log_scale_clamp = 5.0

	def f(self, x):
		'''
		x: [B, full_dim]
		---
		z: [B, full_dim]
		'''
		if self.change_first:
			net_input = x[:, self.half_dim:] # input second half
		else:
			net_input = x[:, :self.half_dim] # input first half
		net_out = self.net(net_input)
		log_scale = net_out[:, :self.half_dim]
		log_scale = torch.clamp(log_scale, -self.log_scale_clamp, self.log_scale_clamp) # avoid exp -> inf
		bias = net_out[:, self.half_dim:]
		if self.change_first:
			modified = torch.exp(log_scale) * x[:, :self.half_dim] + bias
			z = torch.cat([modified, net_input], dim=1)
		else:
			modified = torch.exp(log_scale) * x[:, self.half_dim:] + bias
			z = torch.cat([net_input, modified], dim=1)
		return z

	def f_inv(self, z):
		'''
		z: [B, full_dim]
		---
		x: [B, full_dim]
		'''
		if self.change_first:
			net_input = z[:, self.half_dim:] # input second half (unchanged)
		else:
			net_input = z[:, :self.half_dim] # input first half (unchanged)
		net_out = self.net(net_input)
		log_scale = net_out[:, :self.half_dim]
		log_scale = torch.clamp(log_scale, -self.log_scale_clamp, self.log_scale_clamp) # avoid exp -> inf
		bias = net_out[:, self.half_dim:]
		if self.change_first:
			modified = torch.exp(-log_scale) * (z[:, :self.half_dim] - bias)
			x = torch.cat([modified, net_input], dim=1)
		else:
			modified = torch.exp(-log_scale) * (z[:, self.half_dim:] - bias)
			x = torch.cat([net_input, modified], dim=1)
		return x

	def log_det_jac(self, x):
		'''
		x: [B, full_dim]
		---
		res: [B]
		'''
		if self.change_first:
			net_input = x[:, self.half_dim:] # input second half
		else:
			net_input = x[:, :self.half_dim] # input first half
		net_out = self.net(net_input)
		log_scale = net_out[:, :self.half_dim]
		log_scale = torch.clamp(log_scale, -self.log_scale_clamp, self.log_scale_clamp) # avoid exp -> inf
		res = log_scale.sum(dim=1) # log(det(diag(exp(ls))))=log(prod(exp(ls)))=sum(log(exp(ls)))=sum(ls)
		return res


class AdditiveCoupling1D(Flow):
	def __init__(self, full_dim, change_first):
		super(AdditiveCoupling1D, self).__init__()
		self.change_first = change_first
		self.half_dim = full_dim//2
		hidden_dim = 1000
		self.net = nn.Sequential(
			nn.Linear(self.half_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),	
			nn.Linear(hidden_dim, self.half_dim)
		)

	def f(self, x):
		'''
		x: [B, full_dim]
		---
		z: [B, full_dim]
		'''
		if self.change_first:
			net_input = x[:, self.half_dim:] # input second half
		else:
			net_input = x[:, :self.half_dim] # input first half
		net_out = self.net(net_input)
		if self.change_first:
			modified = x[:, :self.half_dim] + net_out
			z = torch.cat([modified, net_input], dim=1)
		else:
			modified = x[:, self.half_dim:] + net_out
			z = torch.cat([net_input, modified], dim=1)
		return z

	def f_inv(self, z):
		'''
		z: [B, full_dim]
		---
		x: [B, full_dim]
		'''
		if self.change_first:
			net_input = z[:, self.half_dim:] # input second half (unchanged)
		else:
			net_input = z[:, :self.half_dim] # input first half (unchanged)
		net_out = self.net(net_input)
		if self.change_first:
			modified = z[:, :self.half_dim] - net_out
			x = torch.cat([modified, net_input], dim=1)
		else:
			modified = z[:, self.half_dim:] - net_out
			x = torch.cat([net_input, modified], dim=1)
		return x

	def log_det_jac(self, x):
		'''
		x: [B, full_dim]
		---
		res: [B]
		'''
		B = x.size(0)
		res = torch.zeros(B, device=device)
		return res

class DiagPosScaling(Flow):
	def __init__(self, full_dim):
		super(DiagPosScaling, self).__init__()
		self.params = nn.Parameter(torch.zeros(1, full_dim))

	def f(self, x):
		'''
		x: [B, full_dim]
		---
		z: [B, full_dim]
		'''
		z = torch.exp(self.params)*x
		return z

	def f_inv(self, z):
		'''
		z: [B, full_dim]
		---
		x: [B, full_dim]
		'''
		x = torch.exp(-self.params)*z
		return x

	def log_det_jac(self, x):
		'''
		x: [B, full_dim]
		---
		res: [B]
		'''
		B = x.size(0)
		ldj = torch.sum(self.params, dim=1)
		res = ldj.expand((B,))
		return res

def image_to_evenodd(x2d):
	'''
	x2d: [B, 1, S, S]
	---
	x_flat: [B, S*S] (even/odd positions in first/second halves)
	'''
	B, _, S, _ = x2d.shape
	sep = x2d.view(B, S*S//2, 2)
	x_flat = torch.cat([sep[:, :, 0], sep[:, :, 1]], dim=1)
	return x_flat

def evenodd_to_image(x_flat):
	'''
	x_flat: [B, S*S] (even/odd positions in first/second halves)
	---
	x2d: [B, 1, S, S]
	'''
	B, Ssq = x_flat.shape
	S = int(np.sqrt(Ssq)+1e-3)
	sep = torch.stack([x_flat[:, :Ssq//2], x_flat[:, Ssq//2:]], dim=2)
	x2d = sep.view((B, 1, S, S))
	return x2d

def log_pdf_unitnormal(z):
	'''
	z: [B, D]
	---
	res: [B]
	'''
	elewise = -0.5*z**2 - 0.5*np.log(2*np.pi)
	res = elewise.sum(dim=1)
	return res

def log_pdf_logistic(z):
	'''
	z: [B, D]
	---
	res: [B]
	'''
	z = torch.clamp(z, -10., 10.) # avoid exp -> inf
	elewise = -torch.log(1.0+torch.exp(z))-torch.log(1.0+torch.exp(-z))
	res = elewise.sum(dim=1)
	return res

def make_nice(full_dim):
	flow = CompositeFlow([
		AdditiveCoupling1D(full_dim, True),
		AdditiveCoupling1D(full_dim, False),
		AdditiveCoupling1D(full_dim, True),
		AdditiveCoupling1D(full_dim, False),
		DiagPosScaling(full_dim)
	])
	return flow

def make_affine(full_dim):
	flow = CompositeFlow([
		AffineCoupling1D(full_dim, True),
		AffineCoupling1D(full_dim, False),
		AffineCoupling1D(full_dim, True),
		AffineCoupling1D(full_dim, False),
		AffineCoupling1D(full_dim, True),
		AffineCoupling1D(full_dim, False),
		AffineCoupling1D(full_dim, True),
		AffineCoupling1D(full_dim, False),
		DiagPosScaling(full_dim)
	])
	return flow

def test3():
	full_dim = 28*28
	flow = make_affine(full_dim)
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor()
	])
	ds = torchvision.datasets.MNIST('./data', download=True, transform=transform)
	B = 32
	optimizer = torch.optim.Adam(flow.parameters(), lr=0.00001)
	loader = DataLoader(ds, batch_size=B)
	flow.to(device)
	#flow.load_state_dict(torch.load('data/state_dict/sd_8'))
	flow.train()
	n_epochs = 1000
	for epoch in range(1, n_epochs+1):
		running_n = 0
		running_loss = 0.0
		for batch in loader:
			x2d, _ = batch # [B, 1, 28, 28]
			B = x2d.size(0)
			noise = (1.0/256)*torch.rand(x2d.shape)
			x2d = x2d + noise
			optimizer.zero_grad()
			x2d = x2d.to(device)
			x_flat = image_to_evenodd(x2d)
			log_det_jac = flow.log_det_jac(x_flat)
			z = flow.f(x_flat) # TODO: factor out redundant computation
			log_p_z = log_pdf_logistic(z)
			log_p_x = log_p_z + log_det_jac # change of variables
			loss = -log_p_x.mean() # minimize NLL
			if torch.isinf(loss):
				print("Got infinity!")
				torch.save(flow.state_dict(), 'data/bad_sd')
				torch.save(x2d, 'data/bad_x2d')
				return
			if torch.isnan(loss):
				print("Got Nan!")
				torch.save(flow.state_dict(), 'data/bad_sd')
				torch.save(x2d, 'data/bad_x2d')
				return
			loss.backward()
			grad_norm = nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
			optimizer.step()
			running_n += B
			running_loss += B*loss.item()
			print("loss: {:g}, grad_norm: {:g}".format(loss.item(), grad_norm))
		avg_log_p_x = -(running_loss/running_n)
		print('Epoch {:d} avg log p(x): {:g}'.format(epoch, avg_log_p_x))
		torch.save(flow.state_dict(), 'data/state_dict/sd_{:d}'.format(epoch))
		torch.save(flow.state_dict(), 'data/state_dict/sd_latest'.format(epoch))

def test4():
	full_dim = 28*28
	flow = make_affine(full_dim)
	flow.load_state_dict(torch.load('data/state_dict/sd_latest'))
	flow.to(device)
	flow.eval()
	with torch.no_grad():
		for _ in range(10):
			z = torch.randn((1, full_dim), device=device)
			x_flat = flow.f_inv(z)
			log_p_z = log_pdf_logistic(z)
			print(log_p_z)
			z_recon = flow.f(x_flat)
			print(torch.sum(torch.abs(z_recon-z)))
			log_det_jac = flow.log_det_jac(x_flat)
			print(log_det_jac)
			x2d = evenodd_to_image(x_flat)
			sample = x2d[0, 0, :, :].cpu().numpy()
			plt.imshow(sample)
			plt.show()

def test5():
	full_dim = 28*28
	flow = make_affine(full_dim).to(device)
	flow.load_state_dict(torch.load('data/bad_sd'))
	x2d = torch.load('data/bad_x2d').to(device)
	print(x2d.shape)
	zero_inp = torch.zeros(x2d.shape, device=device)
	x_flat = image_to_evenodd(x2d)
	z = flow.f(x_flat)
	log_p_z = log_pdf_logistic(z)
	ldj = flow.log_det_jac(x_flat)
	print(z)
	print(ldj)
	print(log_p_z)
	print(z.max())
	print(z.min())
	pass

test4()