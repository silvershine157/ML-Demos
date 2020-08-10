import torch
import torch.nn as nn
import numpy as np
from scipy.stats import special_ortho_group

try_gpu = False
device = torch.device("cuda" if (try_gpu and torch.cuda.is_available()) else "cpu")

class CouplingNN(nn.Module):

	def __init__(self, C_half):
		super(CouplingNN, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(C_half, 512, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 1),
			nn.ReLU(),
			nn.Conv2d(512, C_half, 3, padding=1)
		)

	def forward(self, nn_in):
		'''
		nn_in: [B, C_half, S, S]
		---
		nn_out: [B, C_half, S, S]
		'''
		nn_out = self.layers(nn_in)
		return nn_out

class StepOfFlow(nn.Module):

	def __init__(self, C):
		super(StepOfFlow, self).__init__()
		self.actnorm_scale = nn.Parameter(torch.ones([1, C, 1, 1]))
		self.actnorm_bias = nn.Parameter(torch.zeros([1, C, 1, 1]))
		W = torch.Tensor(special_ortho_group.rvs(C)) # random C-dim rotation matrix
		self.conv_W = nn.Parameter(W)
		assert C%2 == 0
		self.coupling_nn = CouplingNN(C//2)

	def forward_flow(self, h_in):
		'''
		h_in: [B, C, S, S]
		---
		h_out: [B, C, S, S]
		log_det_jac: [B]
		'''
		B, C, S, _ = h_in.shape
		h_actnorm = self.actnorm_scale * h_in + self.actnorm_bias
		ldj_actnorm = S*S*torch.sum(torch.log(torch.abs(self.actnorm_scale))).expand((B))
		h_conv = torch.einsum('ji,bihw->bjhw', self.conv_W, h_actnorm)
		ldj_conv = S*S*torch.log(torch.abs(torch.det(self.conv_W))).expand((B))
		first_half = h_conv[:, :C//2, :, :]
		h_out = h_conv.clone()
		h_out[:, C//2:, :, :] += self.coupling_nn(first_half)
		log_det_jac = ldj_actnorm + ldj_conv # additive coupling is volume preserving
		return h_out, log_det_jac

	def inverse_flow(self, h_out):
		'''
		h_out: [B, C, S, S]
		---
		h_in: [B, C, S, S]
		'''
		B, C, S, _ = h_out.shape
		first_half = h_out[:, :C//2, :, :]
		h_conv = h_out.clone()
		h_conv[:, C//2:, :, :] -= self.coupling_nn(first_half)
		inv_W = self.conv_W.inverse()
		h_actnorm = torch.einsum('ji,bihw->bjhw', inv_W, h_conv)
		h_in = (h_actnorm - self.actnorm_bias)/self.actnorm_scale
		return h_in

def squeeze(h_in):
	'''
	h_in: [B, C, S, S]
	---
	h_out: [B, 4*C, S//2, S//2]
	'''
	B, C, S, _ = h_in.shape
	assert S%2 == 0
	temp = h_in.view(B, C, S//2, 2, S//2, 2)
	temp = temp.transpose(2, 3) # [B, C, 2, S//2, S//2, 2]
	temp = temp.transpose(4, 5) # [B, C, 2, S//2, 2, S//2]
	temp = temp.transpose(3, 4) # [B, C, 2, 2, S//2, S//2]
	h_out = temp.contiguous().view(B, 4*C, S//2, S//2)
	return h_out

def unsqueeze(h_out):
	'''
	h_out: [B, 4*C, S//2, S//2]
	---
	h_in: [B, C, S, S]
	'''
	B, C_x4, S_half, _ = h_out.shape
	C = C_x4//4
	S = 2*S_half
	temp = h_out.view(B, C, 2, 2, S_half, S_half)
	temp = temp.transpose(3, 4)
	temp = temp.transpose(4, 5)
	temp = temp.transpose(2, 3) # [B, C, S//2, 2, S//2, 2]
	h_in = temp.contiguous().view(B, C, S, S)
	return h_in

class Glow(nn.Module):

	def __init__(self, K, L, C):
		super(Glow, self).__init__()
		self.flow_steps = nn.ModuleList(
			[StepOfFlow(4*C) for _ in range(K)] # after squeeze (x4)
		)
		self.C = C
		self.K = K
		if L > 1:
			self.sub_glow = Glow(K, L-1, C*2) # after split (/2)
		else:
			self.sub_glow = None

	def forward_flow(self, x):
		'''
		x: [B, C, S, S]
		---
		z: [B, C * S * S]
		log_det_jac: [B]
		'''
		B, C, _, _ = x.shape
		h = squeeze(x)
		current_ldj = torch.zeros((B), device=device)
		for k in range(self.K):
			h, step_ldj = self.flow_steps[k].forward_flow(h)
			current_ldj += step_ldj
		if self.sub_glow is not None:
			current_z = h[:, :2*C, :, :].contiguous().view(B, -1)
			x_prop = h[:, 2*C:, :, :]
			sub_z, sub_ldj = self.sub_glow.forward_flow(x_prop)
			z = torch.cat([current_z, sub_z], dim=1)
			log_det_jac = current_ldj + sub_ldj
		else:
			z = h.contiguous().view(B, -1)
			log_det_jac = current_ldj
		return z, log_det_jac

	def inverse_flow(self, z):
		'''
		z: [B, C * S * S]
		---
		x: [B, C, S, S]
		'''
		B, flat_size = z.shape
		C = self.C
		S = int(np.sqrt(flat_size//C)+0.1)
		if self.sub_glow is not None:
			current_z = z[:, :flat_size//2]
			sub_z = z[:, flat_size//2:]
			x_prop = self.sub_glow.inverse_flow(sub_z)
			h_first = current_z.view(B, 2*C, S//2, S//2)
			h = torch.cat([h_first, x_prop], dim=1) # [B, 4*C, S//2, S//2]
		else:
			h = z.view(B, 4*C, S//2, S//2)
		for k in range(self.K):
			h = self.flow_steps[self.K - k - 1].inverse_flow(h)
		x = unsqueeze(h)
		return x

	def log_likelihood(self, x):
		'''
		x: [B, C, S, S]
		---
		LL: [B]
		'''
		z, log_det_jac = self.forward_flow(x)
		logPz = log_pdf_unitnormal(z)
		LL = logPz + log_det_jac
		return LL

def log_pdf_unitnormal(z):
	'''
	z: [B, D]
	---
	res: [B]
	'''
	elewise = -0.5*z**2 - 0.5*np.log(2*np.pi)
	res = elewise.sum(dim=1)
	return res
