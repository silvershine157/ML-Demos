import torch

# Glow with additive coupling layer & no LU decomposition

def additive_coupling(h_in):
	'''
	h_in: [B, S, S, C]
	---
	h_out: [B, S, S, C]
	'''

	def NN(nn_in):
		'''
		nn_in: [B, S, S, C//2]
		---
		nn_out: [B, S, S, C//2]
		'''
		nn_out = nn_in # TODO: use 3 layer CNN (parametrized)
		return nn_out

	_, S, _, C = h_in.shape
	assert C%2 == 0
	first_half = h_in[:, :, :, :C//2]
	nn_out = NN(first_half)
	h_out = h_in.clone()
	h_out[:, :, :, C//2:] += nn_out
	return h_out

def invertible_1x1_conv(h_in):
	'''
	h_in: [B, S, S, C]
	---
	h_out: [B, S, S, C]
	'''
	_, _, _, C = h_in.shape
	W = torch.randn(C, C) # param
	h_out = torch.einsum('ij,bhwi->bhwj', W, h_in)
	return h_out

def actnorm(h_in):
	'''
	h_in: [B, S, S, C]
	---
	h_out: [B, S, S, C]
	'''
	_, _, _, C = h_in.shape
	scale = torch.randn(1, 1, 1, C) # param
	bias = torch.randn(1, 1, 1, C) # param
	h_out = scale * h_in + bias # broadcasting
	return h_out

def step_of_flow(h_in):
	'''
	h_in: [B, S, S, C]
	---
	h_out: [B, S, S, C]
	'''
	h1 = actnorm(h_in)
	h2 = invertible_1x1_conv(h1)
	h_out = additive_coupling(h2)
	return h_out

def test1():
	h_in = torch.randn(2, 4, 4, 6)
	h_out = step_of_flow(h_in)
	print(h_out.shape)

test1()