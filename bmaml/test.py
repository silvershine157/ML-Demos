import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal

def generate_task(return_params=False):
	# sample ground truth parameters from a distribution
	# learning this distribution is the objective of meta learning
	# recommended config for interesting posteriors: w = 1.0, b = 2.0, N = 5

	w = np.random.normal(0.0, 4.0)
	b = np.random.normal(0.0, 4.0)
	# TODO: some interesting regularity to meta-learn (bimodal, sharp reject etc.)
	
	params = torch.FloatTensor([w, b])
	N = 10
	D_train = generate_task_data(N, params)
	D_val = generate_task_data(N, params)
	if not return_params:
		return D_train, D_val
	else:
		return D_train, D_val, params

def generate_task_data(n_samples, params):
	cnt_c0 = 0
	cnt_c1 = 0
	X = torch.zeros((n_samples*2, 1))
	Y = torch.zeros((n_samples*2, 1))
	# this is better than X|Y since the true Y|X is in our model space
	while (min(cnt_c0, cnt_c1) < n_samples):
		x = torch.FloatTensor(np.random.normal(0.0, 5.0, [1, 1]))
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
	optimizer = torch.optim.SGD([params], lr=0.1)
	n_epochs = 5000
	for epoch in range(n_epochs):
		negLL = -avgLL(params, D_train)
		optimizer.zero_grad()
		negLL.backward()
		optimizer.step()
	params = params.detach()
	return params

def model(X, params):
	w, b = params
	return 1.0/(1.0 + torch.exp(-w*(X-b)))

def test_vanila():
	D_train, D_val = generate_task()
	initial_params = torch.FloatTensor([0.0, 0.0])
	new_params = vanila_train(D_train, initial_params)
	print(new_params)

def log_prior(params):
	m = MultivariateNormal(torch.zeros(2), 25.0*torch.eye(2))
	return m.log_prob(params)

def unnorm_log_posterior(params, D):
	X, Y = D
	n_samples = X.size(0)
	log_likelihood = n_samples*avgLL(params, D)
	return log_likelihood #+ log_prior(params)

def test_posterior_sampling():
	# estimate posterior using samples 
	D_train, D_val = generate_task()
	# rejection sampling
	q = MultivariateNormal(torch.zeros(2), 9.0*torch.eye(2))
	k = 0.5
	samples = []
	for i in range(5000):
		z = q.sample()
		u = (k*torch.exp(q.log_prob(z)))*np.random.rand()
		p_z = torch.exp(unnorm_log_posterior(z, D_train))
		if u < p_z:
			samples.append(z)
		else:
			continue
	samples = torch.stack(samples).numpy()
	plt.scatter(samples[:, 0], samples[:, 1])
	plt.show()

def test_posterior_grid():
	D_train, D_val = generate_task()
	# evaluate unnormalized posterior
	n_grid = 30
	grid_size = 10.
	w_grid = np.linspace(-grid_size, grid_size, n_grid)
	b_grid = np.linspace(-grid_size, grid_size, n_grid)
	ulp_values = np.zeros((n_grid, n_grid))
	for w_i, w in enumerate(w_grid):
		for b_i, b in enumerate(b_grid):
			params = torch.Tensor([w, b])
			ulp_values[w_i, b_i] = unnorm_log_posterior(params, D_train)
	up_values = np.exp(ulp_values) # correct up to multiplication constant

	res_img = np.transpose(up_values)
	res_img = np.flip(res_img, axis=0)
	plt.imshow(res_img, interpolation='bicubic', extent=[-grid_size, grid_size, -grid_size, grid_size])
	plt.show()


def test_maml():
	train_tasks = [generate_task() for _ in range(10)]
	val_tasks = [generate_task() for _ in range(10)]


def test_loss_grid():
	D_train, D_val = generate_task()
	n_grid = 30
	grid_size = 10.
	w_grid = np.linspace(-grid_size, grid_size, n_grid)
	b_grid = np.linspace(-grid_size, grid_size, n_grid)
	nll_values = np.zeros((n_grid, n_grid))
	for w_i, w in enumerate(w_grid):
		for b_i, b in enumerate(b_grid):
			params = torch.Tensor([w, b])
			nll_values[w_i, b_i] = -avgLL(params, D_train)

	res_img = np.log(nll_values)
	res_img = np.transpose(res_img)
	res_img = np.flip(res_img, axis=0)
	plt.imshow(res_img, interpolation='bicubic', extent=[-grid_size, grid_size, -grid_size, grid_size])
	plt.show()

def test_ml_bayes_gt():
	D_train, D_val, true_params = generate_task(return_params=True)
	#initial_params = torch.FloatTensor([0.0, 0.0])
	#new_params = vanila_train(D_train, initial_params)

	n_grid = 30
	grid_size = 10.
	w_grid = np.linspace(-grid_size, grid_size, n_grid)
	b_grid = np.linspace(-grid_size, grid_size, n_grid)
	ulp_values = np.zeros((n_grid, n_grid))
	for w_i, w in enumerate(w_grid):
		for b_i, b in enumerate(b_grid):
			params = torch.Tensor([w, b])
			ulp_values[w_i, b_i] = unnorm_log_posterior(params, D_train)
	up_values = np.exp(ulp_values) # correct up to multiplication constant

	res_img = np.transpose(up_values)
	res_img = np.flip(res_img, axis=0)
	plt.imshow(res_img, interpolation='bicubic', extent=[-grid_size, grid_size, -grid_size, grid_size])
	
	true_w, true_b = true_params
	ml_w, ml_b = vanila_train(D_train, true_params)

	plt.plot(true_w, true_b, 'ro')
	plt.plot(ml_w, ml_b, 'bo')
	plt.show()

test_ml_bayes_gt()