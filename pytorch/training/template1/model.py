import torch
import torch.nn as nn

class Net(nn.Module):
	
	def __init__(self):
		super(Net, self).__init__()
		self.layers=nn.Sequential(
			nn.Linear(28*28, 1000),
			nn.ReLU(),
			nn.Linear(1000, 10),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		"""
		input:
			x: gray scale image [B, 28, 28]
		output:
			y_h: label distribution [B, 10]
		"""
		B = x.size(0)
		x = x.view((B, -1))
		y_h = self.layers(x)
		return y_h
