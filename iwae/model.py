import torch
import torch.nn as nn

class VAE(nn.Module):
	def __init__(self, Dx, Dz):
		super(VAE, self).__init__()
		self.enc = Encoder(Dx, Dz)
		self.dec = Decoder(Dx, Dz)
		self.Dz = Dz
		self.Dx = Dx

	def loss(self, x):
		'''
		x: [B, Dx]
		---
		ELBO: float
		'''
		B = x.size(0)
		mean, var = self.enc(x)
		eps = torch.randn(B, self.Dz)
		z = mean + torch.sqrt(var) * eps # reparametrization
		x_r = self.dec(z)
		LL = nn.functional.binary_cross_entropy(x_r, x, reduction='mean')
		KLD = unitGaussianKLD(mean, var)
		ELBO = self.Dx*LL - self.Dz*KLD # TODO: Is this correct?
		return ELBO

def unitGaussianKLD(mean, var):
	'''
	mean: [B, Dz]
	var: [B, Dz]
	---
	KLD: float  (= KLD[N(mean, var) || N(0, I)])
	'''
	KLD = 0.5*torch.mean(1. + torch.log(var) - mean**2 - var)
	return KLD

class Encoder(nn.Module):
	def __init__(self, Dx, Dz):
		super(Encoder, self).__init__()
		self.layers=nn.Sequential(
			nn.Linear(Dx, 100),
			nn.Tanh(),
			nn.Linear(100, Dz*2)
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
			nn.Linear(Dz, 100),
			nn.Tanh(),
			nn.Linear(100, Dx),
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

