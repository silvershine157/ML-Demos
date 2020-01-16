import torch
import torch.nn as nn
import numpy as np
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
		log_p_x_given_z = bernoulliLL(x, x_r)
		KLD = unitGaussianKLD(mean, var)
		ELBO = log_p_x_given_z - KLD
		loss = -torch.mean(ELBO) # minimization objective for batch
		return loss

	def loss_alternate(self, x2d):
		# no KLD
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
		unit_mean = torch.zeros(B, self.Dz, device=device)
		unit_var =  torch.ones(B, self.Dz, device=device)
		log_p_z = factorizedGaussianLogPdf(z, unit_mean, unit_var)
		log_p_x_given_z = bernoulliLL(x, x_r)
		log_p_x_z = log_p_z + log_p_x_given_z
		log_q_z_given_x = factorizedGaussianLogPdf(z, mean, var)
		ELBO = log_p_x_z - log_q_z_given_x
		loss = -torch.mean(ELBO) # minimization objective for batch
		return loss

	def iwae_loss(self, x2d):
		'''
		x2d: [B, H, W]
		---
		loss: float
		'''
		K = 1 # number of samples
		B, H, W = x2d.size()
		x = x2d.view(B, H*W)
		mean, var = self.enc(x)
		unit_mean = torch.zeros(B, self.Dz, device=device)
		unit_var =  torch.ones(B, self.Dz, device=device)
		all_log_frac_p_q = torch.zeros((B, K), device=device)
		for k in range(K):
			eps = torch.randn((B, self.Dz), device=device)
			z = mean + torch.sqrt(var+1E-7) * eps # reparametrization
			x_r = self.dec(z)
			log_p_z = factorizedGaussianLogPdf(z, unit_mean, unit_var)
			log_p_x_given_z = bernoulliLL(x, x_r)
			log_p_x_z = log_p_z + log_p_x_given_z
			log_q_z_given_x = factorizedGaussianLogPdf(z, mean, var)
			all_log_frac_p_q[:, k] = log_p_x_z - log_q_z_given_x
		log_rscl_factors = torch.max(all_log_frac_p_q, dim=1, keepdim=True)[0]
		all_frac_p_q_rscl = torch.exp(all_log_frac_p_q - log_rscl_factors) # broadcasting
		obj = torch.log(torch.mean(all_frac_p_q_rscl,dim=1)+1E-7)+log_rscl_factors
		loss = -torch.mean(obj) # minimization objective for batch
		return loss

def bernoulliLL(x, x_r):
	'''
	x: [B, Dx]
	x_r: [B, Dx]
	---
	LL: [B]
	'''
	LL = torch.sum(x*torch.log(x_r)+(1-x)*torch.log(1-x_r), dim=1)
	return LL

def unitGaussianKLD(mean, var):
	'''
	mean: [B, Dz]
	var: [B, Dz]
	---
	KLD: [B] (= KLD[N(mean, var) || N(0, I)])
	'''
	KLD = -0.5*torch.sum(1. + torch.log(var+1E-7) - mean**2 - var, dim=1)
	return KLD

def factorizedGaussianLogPdf(z, mean, var):
	'''
	z: [B, Dz]
	mean: [B, Dz]
	var: [B, Dz]
	---
	log_p: [B]
	'''
	log_p_each = -0.5*((z-mean)**2/(var+1E-7)) - torch.log(torch.sqrt(2*np.pi*var+1E-7))
	log_p = torch.sum(log_p_each, dim=1)
	return log_p

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

