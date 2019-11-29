import torch
import torch.nn as nn

class MLP(nn.Module):
	# simple 2 layer MLP
	def __init__(self, d_input):
		super(MLP, self).__init__()
		d_hidden = 3*(d_input+3)
		self.layers=nn.Sequential(
			nn.Linear(d_input, d_hidden),
			nn.ReLU(),
			nn.Linear(d_hidden, 1)
		)

	def forward(self, x):
		"""
		input:
			x: input feature [B, d_input]
		output:
			y_h: prediction [B, 1]
		"""
		y_h = self.layers(x)
		return y_h
