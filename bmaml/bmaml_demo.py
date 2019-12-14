import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from torch.autograd import grad

# Stein variational gradient descent (diff'able w.r.t. init_particles)
def svgd(init_particles, D_train, n_steps):
	'''
	init_particles: [N_particles, D_param]
	D_train:
		X_train: [B_data, D_data]
		Y_train: [B_data, D_out]
	n_steps: int
	new_particles: [N_particles, D_param]
	'''
	return None

# gradient descent (diff'able w.r.t. init_param)
def grad_desc(init_param, D_train, n_steps):
	'''
	init_param: [1, D_param]
	D_train:
		X_train: [B_data, D_data]
		Y_train: [B_data, D_out]
	n_steps: int
	new_param: [1, D_param]
	'''
	return None

# parametrized model
def model(X, param):
	'''
	X: [B_data, D_data]
	param: [B_param, D_param]
	out: [B_param, B_data, D_out]
	'''
	X_ = X.unsqueeze(dim=0).squeeze(dim=2) # [1, B_data]
	b = param[:, 1].view((-1, 1)) # [B_param, 1]
	w = param[:, 0].view((-1, 1)) # [B_param, 1]
	logits = (w*(X_ - b)).unsqueeze(dim=2) # [B_param, B_data, 1]
	out = 1.0/(1.0 + torch.exp(-logits))
	return out

def generate_task_data(B_data, param):
	'''
	D:
		X: [B_data, D_data]
		Y: [B_data, D_out]
	'''
	cnt_c0 = 0
	cnt_c1 = 0
	X = torch.zeros((B_data, 1))
	Y = torch.zeros((B_data, 1))
	B_c0 = B_data//2
	B_c1 = B_data - B_c0
	while (cnt_c0 < B_c0 or cnt_c1 < B_c1):
		x = torch.FloatTensor(np.random.normal(0.0, 5.0, [1, 1]))
		prob = model(x, param)
		if np.random.rand() < prob:
			if cnt_c1 < B_c1:
				cnt_c1 += 1
				y = 1
		else:
			if cnt_c0 < B_c0:
				cnt_c0 += 1
				y = 0
		if y is not None:
			top = cnt_c1 + cnt_c0 - 1
			X[top] = x
			Y[top] = y
	D = (X, Y)
	return D

# avg log likelihood
def avg_log_p_D_given_param(D, param):
	'''
	D:
		X: [B_data, D_data]
		Y: [B_data, D_out]
	param: [B_param, D_param]
	out: [B_param]	
	'''
	return None

# log prior over parameters
def log_p_param(param):
	'''
	param: [B_param, D_param]
	out: [B_param]
	'''
	return None

# log posterior over parameters given data
# unnormalied (correct up to constant addition)
def log_p_param_given_D_unnorm(D, param):
	'''
	X: [B_data, D_data]
	param: [B_param, D_param]
	out: [B_data, B_param]
	'''
	return None

def test():
	B_data = 10
	param = torch.ones((1, 2))
	generate_task_data(B_data, param)
	pass

def demo_1():
	# sample a task with some seed
	# find ML solution using GD
	# compute exact posterior
	# plot [posterior + ML point + GT point] with [data + ML curve + GT curve]
	pass

def demo_2():
	# sample meta-learning data with some seed
	# find init points using MAML
	# compare GT points and adaptation points
	pass

def demo_3():
	# sample a task with some seed
	# compute exact posterior
	# apply SVGD
	# plot exact posterior and SVGD samples
	pass

def demo_4():
	# BFA
	# sample meta-learning data with some seed
	pass

def demo_5():
	# BMAML
	# sample meta-learning data with some seed
	pass

test()