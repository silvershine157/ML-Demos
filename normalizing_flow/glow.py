import torch
import torch.nn as nn


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
		assert C%2 == 0
		self.coupling_nn = CouplingNN(C//2)
		self.actnorm_scale = nn.Parameter(torch.randn([1, C, 1, 1]))
		self.actnorm_bias = nn.Parameter(torch.randn([1, C, 1, 1]))
		self.invertible_1x1_conv_W = nn.Parameter(torch.randn([C, C]))

	def forward_flow(self, h_in):
		'''
		h_in: [B, C, S, S]
		---
		h_out: [B, C, S, S]
		'''
		pass

	def inverse_flow(self, h_out):
		'''
		h_out: [B, C, S, S]
		---
		h_in: [B, C, S, S]
		'''
		pass

	def log_det_jac(self, h_in):
		'''
		h_in: [B, C, S, S]
		---
		ldj: [B, 1]
		'''
		pass


class Glow(nn.Module):

	def __init__(self, K, L, C):
		super(Glow, self).__init__()
		self.flow_steps = nn.ModuleList(
			[StepOfFlow(4*C) for _ in range(K)] # after squeeze (x4)
		)
		if L > 1:
			self.sub_glow = Glow(K, L-1, C*2) # after split (/2)
		else:
			self.sub_glow = None

	def forward_flow(self, x):
		'''
		x: [B, C, S, S]
		---
		z: [B, C * S * S]
		'''
		pass

	def inverse_flow(self, z):
		'''
		z: [B, C * S * S]
		---
		x: [B, C, S, S]
		'''
		pass

	def log_det_jac(self, x):
		'''
		x: [B, C, S, S]
		---
		ldj: [B, 1]
		'''
		pass


def test1():
	C = 1
	K = 10
	L = 3
	glow = Glow(K, L, C)
	print(len(glow.parameters()))
	pass

test1()
