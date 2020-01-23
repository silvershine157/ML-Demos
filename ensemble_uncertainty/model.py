import torch
import torch.nn as nn
import math

class GaussianMLP(nn.Module):
	def __init__(self, d_in=1, d_out=1, d_hidden=50):
		super(GaussianMLP, self).__init__()
		self.d_out = d_out
		self.layers = nn.Sequential(
			nn.Linear(d_in, d_hidden),
			nn.ReLU(),
			nn.Linear(d_hidden, d_out*2)
		)

	def forward(self, x):
		'''
		x: [B, d_in]
		---
		mu: [B, d_out]
		var: [B, d_out]
		'''
		out = self.layers(x)
		mu = out[:, :self.d_out]
		var = torch.log(1+torch.exp(out[:, self.d_out:])) + 1E-6
		return mu, var

	def loss(self, x, y):
		# NLL loss
		'''
		x: [B, d_in]
		y: [B, d_in]
		---
		L: float
		'''
		if self.d_out != 1:
			raise NotImplementedError
		mu, var = self.forward(x)
		LL = -0.5*(y-mu)**2/var - 0.5*torch.log(2*math.pi*var)
		L = -torch.mean(LL) # NLL
		return L

class BasicMLP(nn.Module):
	def __init__(self, d_in=1, d_out=1, d_hidden=50):
		super(BasicMLP, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(d_in, d_hidden),
			nn.ReLU(),
			nn.Linear(d_hidden, d_out)
		)

	def forward(self, x):
		'''
		x: [B, d_in]
		---
		mu: [B, d_out]
		'''
		mu = self.layers(x)
		return mu

	def loss(self, x, y):
		# MSE loss
		'''
		x: [B, d_in]
		y: [B, d_in]
		---
		L: float
		'''
		mu = self.forward(x)
		L = torch.mean((mu - y)**2)
		return L