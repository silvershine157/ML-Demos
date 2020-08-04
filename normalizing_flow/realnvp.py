import torch

def full_flow(x):
	'''
	x: [B, S, S, C]
	---
	z: [B, S*S*C]
	'''
	z = multiscale(x, depth=3)
	return x

def multiscale(h_in, depth):
	'''
	h_in: [B, S, S, C]
	---
	z: [B, S*S*C]
	'''
	if depth == 1:
		z = final_step_of_flow(h_in)
	else:
		z_current, h_current = interm_step_of_flow(h_in)
		z_rest = multiscale(h_current, depth-1)
		z = torch.cat([z_current, z_rest])
	return z

def interm_step_of_flow(h_in):
	'''
	h_in: [B, S, S, C]
	---
	z: [B, S*S*C/2]
	h_out: [B, S/2, S/2, 2C]
	'''
	pass

def final_step_of_flow(h_in):
	'''
	h_in: [B, S, S, C]
	---
	z: [B, S*S*C]
	'''
	pass

def coupling_checkerboard(h_in, mask_switch):
	'''
	h_in: [B, S, S, C]
	---
	h_out: [B, S, S, C]
	'''
	B, S, _, C = h_in.shape
	if mask_switch:
		mask = get_checkerboard_mask(S)
	else:
		mask = ~get_checkerboard_mask(S)
	h_masked = mask*h_in
	print(h_masked[0, :, :, 0])
	# TODO: conv

def coupling_channelwise(h_in, mask_switch):
	'''
	h_in: [B, S, S, C]
	---
	h_out: [B, S, S, C]
	'''

def squeezing(h_in):
	'''
	h_in: [B, S, S, C]
	---
	h_out: [B, S/2, S/2, 4*C]
	'''
	B, S, _, C = h_in.shape
	temp = h_in.view(B, S//2, 2, S//2, 2, C)
	temp = temp.transpose(2,3).contiguous() # [B, S//2, S//2, 2, 2, C]
	h_out = temp.view(B, S//2, S//2, 4*C)
	return h_out

def get_checkerboard_mask(S):
	'''
	S: int
	---
	mask: [1, S, S, 1]
	'''
	mask = torch.ones((S, S)).bool()
	mask[::2, ::2] = 0
	mask[1::2, 1::2] = 0
	mask = mask.view(1, S, S, 1)
	return mask

def test1():
	h_in = torch.randn((1, 6, 6, 2))
	h_in[0, :, 0, :] = 0
	h_in[0, :, 1, :] = 0
	squeezing(h_in)
	pass

def test2():
	h_in = torch.randn((1, 4, 4, 2))
	coupling_checkerboard(h_in, False)

test2()