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
	init_param: [1, D_param], should require grad
	D_train:
		X_train: [B_data, D_data]
		Y_train: [B_data, D_out]
	n_steps: int
	new_param: [1, D_param]
	'''
	param = init_param
	lr = 0.1
	for _ in range(n_steps):
		train_loss = -avgLL(D_train, param) # [1]
		task_grad = grad(train_loss, param)[0] # [1, D_param]
		param = param - lr*task_grad
		#print(train_loss.item())
	new_param = param
	return new_param

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

def generate_task(true_params=None):
	#w = np.random.normal(0.0, 2.0)
	#b = np.random.normal(0.0, 2.0)
	w = 5.
	b = 0.0
	param = torch.FloatTensor([[w, b]]) # [1, 2]
	N = 1000
	D_train = generate_task_data(N, param)
	D_val = generate_task_data(N, param)
	if true_params:
		true_params[:] = param
	return D_train, D_val

# avg log likelihood
def avgLL(D, param):
	'''
	D:
		X: [B_data, D_data]
		Y: [B_data, D_out]
	param: [B_param, D_param]
	out: [B_param]
	'''
	B_param = param.size(0)
	X, Y = D
	Y_pred = model(X, param) # [B_param, B_data, 1]
	Y_ = Y.unsqueeze(dim=0).expand((B_param, -1, -1))
	probs = torch.zeros_like(Y_pred)
	probs[Y_ > 0.5] = Y_pred[Y_ > 0.5] # probability that Y=1
	probs[Y_ < 0.5] = 1-Y_pred[Y_ < 0.5] # probability that Y=0
	out = torch.mean(torch.log(probs + 1.0E-7), dim=1).squeeze(dim=1)
	return out

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

def sample_param(B_param=1):
	# randomized initial parameter
	# out: [B_param, D_param]
	D_param = 2
	param = 0.1*torch.randn(B_param, D_param)
	param.requires_grad_(True)
	return param

def demo_grad_desc():
	D_train, D_val = generate_task()
	init_param = sample_param()
	test_task(D_val, init_param)
	learned_param = grad_desc(init_param, D_train, n_steps=100)
	test_task(D_val, learned_param)

def demo_maml():
	pass

def demo_svgd():
	pass

def demo_bfa():
	# BFA
	D_meta_train, D_meta_val = generate_meta_task()
	init_particles = train_bfa(D_meta_train)
	test_meta_svgd(D_meta_val, init_particles)

def demo_bmaml():
	# BMAML
	D_meta_train, D_meta_val = generate_meta_task()
	init_particles = train_bmaml(D_meta_train)
	test_meta_svgd(D_meta_val, init_particles)

def generate_meta_task():
	N_tasks = 10
	D_meta_train = [generate_task() for _ in range(N_tasks)]
	D_meta_val = [generate_task() for _ in range(N_tasks)]
	return D_meta_train, D_meta_val

def train_bfa(D_meta):
	init_particles = None
	return init_particles	

def train_bmaml(D_meta):
	init_particles = None
	return init_particles

def test_task(D_val, param):
	# param: [1, D_param]
	LL = avgLL(D_val, param).item()
	X, Y = D_val
	Y_pred = model(X, param).squeeze(dim=0)
	n_correct = torch.sum(Y[Y_pred > 0.5])+torch.sum(1-Y[Y_pred < 0.5])
	n_total = X.size(0)
	accuracy = n_correct.item()/n_total
	print('accuracy: {:.02f}'.format(accuracy))
	print('average LL: ', LL)

def test_meta_svgd(D_meta, init_particles):
	pass

def test_meta_grad_desc(D_meta, param):
	pass


demo_grad_desc()