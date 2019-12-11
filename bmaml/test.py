import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_task():
	# sample ground truth parameters from a distribution
	# learning this distribution is the objective of meta learning
	w = 5.0
	#b = np.random.normal(0.0, 0.3)
	b = 0.0
	params = (w, b)
	N_train = 100
	N_val = 50
	D_train = generate_task_data(N_train, params)
	D_val = generate_task_data(N_val, params)
	return D_train, D_val

def generate_task_data(n_samples, params):
	cnt_c0 = 0
	cnt_c1 = 0
	X = torch.zeros((n_samples*2, 1))
	Y = torch.zeros((n_samples*2, 1))
	# this is better than X|Y since the true Y|X is in our model space
	while (min(cnt_c0, cnt_c1) < n_samples):
		x = torch.FloatTensor(np.random.normal(0.0, 1.0, [1, 1]))
		prob = model(x, params)
		y = None
		if np.random.rand() < prob:
			if cnt_c1 < n_samples:
				cnt_c1 += 1
				y = 1
		else:
			if cnt_c0 < n_samples:
				cnt_c0 += 1
				y = 0
		if y is not None:
			top = cnt_c1 + cnt_c0 - 1
			X[top] = x
			Y[top] = y
	D = (X, Y)
	return D

def avgLL(params, D):
	X, Y = D
	Y_prob = model(X, params)
	eps = 1.0E-7
	LL_c0 = torch.sum(torch.log(1-Y_prob[Y < 0.5] + eps))
	LL_c1 = torch.sum(torch.log(Y_prob[Y > 0.5] + eps))
	n_samples = X.size(0)
	avg_LL_total = (LL_c0+LL_c1)/n_samples
	return avg_LL_total

def vanila_train(D_train, old_params):
	params = old_params.clone()
	params.requires_grad_(True)
	optimizer = torch.optim.SGD([params], lr=1.)
	n_epochs = 1000
	for epoch in range(n_epochs):
		negLL = -avgLL(params, D_train)
		optimizer.zero_grad()
		negLL.backward()
		optimizer.step()
	return params

def model(X, params):
	w, b = params
	return 1.0/(1.0 + torch.exp(-w*(X-b)))

def test():
	D_train, D_val = generate_task()
	initial_params = torch.FloatTensor([-5.0, 1.0])
	new_params = vanila_train(D_train, initial_params)
	print(new_params)

test()