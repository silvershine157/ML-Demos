import torch
from model import GaussianMLP, BasicMLP

def test():
	x = torch.zeros([4, 1])
	gmlp = GaussianMLP()
	bmlp = BasicMLP()
	mu, var = gmlp(x)
	print(mu.shape)
	print(var.shape)
	mu = bmlp(x)
	print(mu.shape)

test()