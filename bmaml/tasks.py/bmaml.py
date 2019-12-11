import torch
import numpy as np

def generate_task():
	# sample w, b
	# guessing this is the objective of meta learning
	w = 5.0
	b = np.random.normal(0.0, 0.3)
	N_train = 10
	N_val = 5
	D_train = generate_task_data(N_train, w, b)
	D_val = generate_task_data(N_val, w, b)
	return D_train, D_val

def generate_task_data(n_samples, w, b):
	cnt_c0 = 0
	cnt_c1 = 0
	X = torch.zeros((n_samples*2, 1))
	Y = torch.zeros((n_samples*2, 1))
	# this is better than X|Y since the true Y|X is in our model space
	while (min(cnt_c0, cnt_c1) < n_samples):
		x = torch.FloatTensor(np.random.normal(0.0, 1.0, [1, 1]))
		prob = model(x, w, b)
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

def model(x, w, b):
	return 1.0/(1.0 + torch.exp(-w*(x-b)))

def train_vanila(D_train, w0, b0):
	# use full data
	w = torch.FloatTensor([w0])
	b = torch.FloatTensor([b0])
	X_train, Y_train = D_train
	Y_pred = model(X_train, w, b)
	print(Y_pred)
	pass

def demo_vanila():
	D_train, D_val = generate_task()
	train_vanila(D_train, 0.0, 0.0)


demo_vanila()