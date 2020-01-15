import torch
import torch.nn as nn
from const import *


class VAE(nn.Module):
	def __init__(self, H, W, Dz):
		super(VAE, self).__init__()
		self.Dx = H*W
		self.Dz = Dz
		self.enc = Encoder(self.Dx, Dz)
		self.dec = Decoder(self.Dx, Dz)

	def loss(self, x2d):
		'''
		x2d: [B, H, W]
		---
		loss: float
		'''
		B, H, W = x2d.size()
		x = x2d.view(B, H*W)
		mean, var = self.enc(x)
		eps = torch.randn((B, self.Dz), device=device)
		z = mean + torch.sqrt(var+1E-7) * eps # reparametrization
		x_r = self.dec(z)
		'''
		print("HIHI")
		print(torch.max(x_r))
		print(torch.max(x))
		print(torch.min(x_r))
		print(torch.min(x))
		'''
		negLL = nn.functional.binary_cross_entropy(x_r, x, reduction='mean')
		KLD = unitGaussianKLD(mean, var)
		ELBO = self.Dx*(-negLL) - self.Dz*KLD # TODO: Is this correct?
		loss = -ELBO
		return loss


def unitGaussianKLD(mean, var):
	'''
	mean: [B, Dz]
	var: [B, Dz]
	---
	KLD: float  (= KLD[N(mean, var) || N(0, I)])
	'''
	KLD = 0.5*torch.mean(1. + torch.log(var+1E-7) - mean**2 - var)
	return KLD

class Encoder(nn.Module):
	def __init__(self, Dx, Dz):
		super(Encoder, self).__init__()
		self.layers=nn.Sequential(
			nn.Linear(Dx, 500),
			nn.Tanh(),
			nn.Linear(500, Dz*2)
		)
		self.Dz = Dz

	def forward(self, x):
		'''
		x: [B, Dx]
		---
		mean: [B, Dz]
		var: [B, Dz]
		'''
		res = self.layers(x)
		mean = torch.tanh(res[:, :self.Dz])
		var = torch.exp(res[:, self.Dz:])
		return mean, var

class Decoder(nn.Module):
	def __init__(self, Dx, Dz):
		super(Decoder, self).__init__()
		self.layers=nn.Sequential(
			nn.Linear(Dz, 500),
			nn.Tanh(),
			nn.Linear(500, Dx),
			nn.Sigmoid()
		)

	def forward(self, z):
		'''
		z: [B, Dz]
		---
		x_r: [B, Dx]
		'''
		x_r = self.layers(z)
		return x_r

