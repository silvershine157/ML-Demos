import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class AffineCouplingLayer(nn.Module):

	def __init__(self, switch):
		super(AffineCouplingLayer, self).__init__()
		d_hidden = 100
		self.coupling_nn = nn.Sequential(
			nn.Linear(1, d_hidden),
			nn.Tanh(),
			nn.Linear(d_hidden, d_hidden),
			nn.Tanh(),
			nn.Linear(d_hidden, 2) # log-scale and bias
		)
		self.switch = switch

	def forward_map(x):
		'''
		x: [B, 2]
		---
		z: [B, 2]
		log_det: [B]
		'''		
		if self.switch:
			nn_in = x[:, 0]
			nn_out = self.coupling_nn(nn_in)
			log_scale = nn_out[:, 0]
			bias = nn_out[:, 1]
			z = x.clone()
			z[:, 1] = torch.exp(log_scale)*x[:, 1]+bias
			log_det = log_scale
		else:
			nn_in = x[:, 1]
			nn_out = self.coupling_nn(nn_in)
			log_scale = nn_out[:, 0]
			bias = nn_out[:, 1]
			z = x.clone()
			z[:, 0] = torch.exp(log_scale)*x[:, 0]+bias
			log_det = log_scale
		return z, log_det

class Flow(nn.Module):

	def __init__(self):
		super(Flow, self).__init__()
		self.K = 4
		self.flow_steps = nn.ModuleList(
			[AffineCouplingLayer(i%2==0) for i in range(self.K)]
		)

	def forward_map(x):
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
			h, step_log_det = self.flow_steps[k]
			log_det += step_log_det
		z = h
		return z, log_det

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
	return data

def test1():
	N = 300
	x = make_data(N)
	plt.scatter(x[:, 0], x[:, 1])
	plt.show()

test1()