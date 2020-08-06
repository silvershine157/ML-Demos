import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class AffineCouplingLayer(nn.Module):

	def __init__(self, switch):
		super(AffineCouplingLayer, self).__init__()
		d_hidden = 128
		self.coupling_nn = nn.Sequential(
			nn.Linear(1, d_hidden),
			nn.Tanh(),
			nn.Linear(d_hidden, d_hidden),
			nn.Tanh(),
			nn.Linear(d_hidden, 2) # log-scale and bias
		)
		self.switch = switch

	def forward_map(self, x):
		'''
		x: [B, 2]
		---
		z: [B, 2]
		log_det: [B]
		'''		
		if self.switch:
			nn_in = x[:, 0].view(-1, 1)
			nn_out = self.coupling_nn(nn_in)
			log_scale = nn_out[:, 0]
			bias = nn_out[:, 1]
			z = x.clone()
			z[:, 1] = torch.exp(log_scale)*x[:, 1]+bias
			log_det = log_scale
		else:
			nn_in = x[:, 1].view(-1, 1)
			nn_out = self.coupling_nn(nn_in)
			log_scale = nn_out[:, 0]
			bias = nn_out[:, 1]
			z = x.clone()
			z[:, 0] = torch.exp(log_scale)*x[:, 0]+bias
			log_det = log_scale
		return z, log_det

	def inverse_map(self, z):
		'''
		z: [B, 2]
		---
		x: [B, 2]
		'''
		if self.switch:
			nn_in = z[:, 0].view(-1, 1)
			nn_out = self.coupling_nn(nn_in)
			log_scale = nn_out[:, 0]
			bias = nn_out[:, 1]
			x = z.clone()
			x[:, 1] = torch.exp(-log_scale)*(x[:, 1]-bias)
		else:
			nn_in = z[:, 1].view(-1, 1)
			nn_out = self.coupling_nn(nn_in)
			log_scale = nn_out[:, 0]
			bias = nn_out[:, 1]
			x = z.clone()
			x[:, 0] = torch.exp(-log_scale)*(x[:, 0]-bias)
		return x

class Flow(nn.Module):

	def __init__(self, K):
		super(Flow, self).__init__()
		self.K = K
		self.flow_steps = nn.ModuleList(
			[AffineCouplingLayer(i%2==0) for i in range(self.K)]
		)

	def forward_map(self, x):
		'''
		x: [B, 2]
		---
		z: [B, 2]
		log_det: [B]
		'''
		h = x
		B = x.shape[0]
		log_det = torch.zeros((B))
		for k in range(self.K):
			h, step_log_det = self.flow_steps[k].forward_map(h)
			log_det += step_log_det
		z = h
		return z, log_det

	def inverse_map(self, z):
		'''
		z: [B, 2]
		---
		x: [B, 2]
		'''
		h = z
		for k in range(self.K):
			h = self.flow_steps[self.K-k-1].inverse_map(h)
		x = h
		return x

def log_pdf_unitnormal(z):
	'''
	z: [B, D]
	---
	res: [B]
	'''
	elewise = -0.5*z**2 - 0.5*np.log(2*np.pi)
	res = elewise.sum(dim=1)
	return res

def make_data(N):
	curve_x = 1.8*(np.random.rand(N//2)-0.5)
	blob1_y = 1.5*curve_x**4 + 0.1*np.random.randn(N//2) - 0.8
	blob1_x = curve_x + 0.5
	curve_x = 1.8*(np.random.rand(N//2)-0.5)
	blob2_y = -1.5*curve_x**4 + 0.1*np.random.randn(N//2) + 0.8
	blob2_x = curve_x - 0.5
	data_x = np.concatenate((blob1_x, blob2_x))
	data_y = np.concatenate((blob1_y, blob2_y))
	data = np.stack((data_x, data_y), axis=1)
	np.random.shuffle(data)
	data = torch.Tensor(data)
	return data

def test1():
	n_iters = 1000
	N = 300
	K = 8
	x = make_data(N)
	flow = Flow(K)
	
	optimizer = torch.optim.Adam(flow.parameters(), lr=0.0001)
	for _ in range(n_iters):
		optimizer.zero_grad()
		z, log_det = flow.forward_map(x)
		LL = log_pdf_unitnormal(z) + log_det
		loss = -LL.mean()
		loss.backward()
		optimizer.step()
		print(loss.item())

	plt.figure()
	plt.subplot(2,2,1)
	show_data(x)
	plt.subplot(2,2,2)
	show_data(z.detach())
	plt.subplot(2,2,4)
	z_sample = torch.randn((N, 2))
	show_data(z_sample)
	plt.subplot(2,2,3)
	x_sample = flow.inverse_map(z_sample)
	show_data(x_sample.detach())
	plt.show()

def show_data(x):
	x = x.numpy()
	plt.scatter(x[:, 0], x[:, 1])

test1()