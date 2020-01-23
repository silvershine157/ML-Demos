import torch
import torch.nn as nn

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


class BasicMLP(nn.Module):
	def __init__(self, d_in=1, d_out=1, d_hidden=50):
		super(BasicMLP, self).__init__()
		self.d_out = d_out
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
