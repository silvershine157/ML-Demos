import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.animation as anim
from scipy.stats import multivariate_normal
import time

## model dimensions
K = 3 # number of latent states
N = 1000 # number of data points
D = 2 # dimension of a data point

# np.random.seed(2)

def generate_data():
	## true model parameters
	# p(z_0): initial distribution
	PI = np.array([0.2, 0.5, 0.3])
	# p(z_n | z_{n-1}): transition matrix
	
	A = np.array([
		[0.85, 0.05, 0.1],
		[0.2, 0.7, 0.1],
		[0.1, 0.05, 0.85]
	])
	# p(x_n | z_n): D-dim gaussian
	PHI_MU = np.array([
		[0.3, 0.7],
		[0.3, 0.2],
		[0.7, 0.5]
	])
	PHI_SIGMA = 0.01*np.array([
		[[0.5, 0.4], [0.4, 0.5]],
		[[0.5, 0.0], [0.0, 0.2]],
		[[0.5, -0.4], [-0.4, 0.5]]
	])
	## generate data
	z = np.zeros((N), dtype=np.int16)
	x = np.zeros((N, D), dtype=np.float32)
	z[0] = np.random.choice(K, p=PI)
	x[0] = np.random.multivariate_normal(PHI_MU[z[0]], PHI_SIGMA[z[0]])
	for n in range(1, N):
		z[n] = np.random.choice(K, p=A[z[n-1], :])
		x[n] = np.random.multivariate_normal(PHI_MU[z[n]], PHI_SIGMA[z[n]])
	return x, z

def visualize_data(x, z=None):
	if z is not None:
		# color code latent variable
		colors = ['red', 'green', 'blue']
		plt.plot(x[:, 0], x[:, 1], color='black', zorder=1)
		plt.scatter(x[:, 0], x[:, 1], 
			c=z, cmap=matplotlib.colors.ListedColormap(colors), zorder=2)
		plt.show()
	else:
		# no color
		plt.plot(x[:, 0], x[:, 1])
		plt.scatter(x[:, 0], x[:, 1])
		plt.show()

def init_params(x):
	pi = np.ones((K))/K
	A = np.ones((K, K))/K

	# singular covariance
	# -> due to collapsing to single datapoint
	# -> improve with K-means initialization
	MU = np.random.random((K, D))
	while True:
		x_diff = np.expand_dims(x, axis=1)-np.expand_dims(MU, axis=0) # (N, K, D)
		sq_dists = np.einsum('nkd,nkd->nk', x_diff, x_diff) # (N, K)
		assign_idx = np.argmin(sq_dists, axis=1)
		assign_onehot = np.eye(K)[assign_idx] # (N, K)
		Nks = np.expand_dims(np.sum(assign_onehot, axis=0), axis=1) # (K, 1)
		new_MU = np.einsum('nd,nk->kd', x, assign_onehot)/(Nks+1)
		change = np.sum(np.abs(MU - new_MU))
		MU = new_MU
		if change < 0.01:
			break

	SIGMA = [0.01*np.eye(D) for _ in range(K)]
	SIGMA = np.stack(SIGMA, axis=0)
	params = (pi, A, MU, SIGMA)
	return params

def em_step(old_params, x):
	
	# E-step
	gamma, xi, avgLL = forward_backward(old_params, x)
	
	# M-step
	pi = gamma[0, :]/np.sum(gamma[0, :])
	A_ = np.sum(xi, axis=0) # (n-1, prev, current)
	A = A_/np.sum(A_, axis=1, keepdims=True) # (prev, current)
	MU = np.zeros((K, D))
	SIGMA = np.zeros((K, D, D))
	for k in range(K):
		gamma_k = np.expand_dims(gamma[:, k], axis=1)
		N_k = np.sum(gamma_k)
		mu_k = np.sum(gamma_k*x,axis=0, keepdims=True)/N_k
		x_c = x-mu_k
		Sigma_k = np.einsum('n,ni,nj->ij', gamma[:, k], x_c, x_c)/N_k
		MU[k, :] = mu_k
		SIGMA[k, :, :] = Sigma_k

	new_params = (pi, A, MU, SIGMA)
	return new_params, avgLL

def forward_backward(params, x):
	'''
	<input>
	params = (pi, A, MU, SIGMA)
		pi: (K)
		A: (K, K) [prev, cur]
		MU: (K, D)
		SIGMA: (K, D, D)
	x: (N, D)
	<output>
	gamma: (N, K)
	xi: (N-1, K, K) [n-1, prev, cur]
	avgLL: scalar
	'''
	pi, A, MU, SIGMA = params
	alpha_ = np.zeros((N, K))
	beta_ = np.zeros((N, K))
	c = np.zeros((N))
	p_x_given_z = np.zeros((N, K)) # can be computed rightaway
	
	# compute p_x_given_z
	for n in range(N):
		for k in range(K):
			p_x_given_z[n, k] = multivariate_normal.pdf(x[n,:], mean=MU[k,:], cov=SIGMA[k,:,:])

	# compute alpha_ (rescaled) and c
	unnorm = pi * p_x_given_z[0, :]
	c[0] = np.sum(unnorm)
	alpha_[0, :] = unnorm/c[0]
	for n in range(1, N):
		temp = np.expand_dims(alpha_[n-1,:], axis=1)*A # element-wise
		unnorm = p_x_given_z[n, :] * np.sum(temp, axis=0)
		c[n] = np.sum(unnorm)
		alpha_[n, :] = unnorm/c[n]

	# compute beta_ (rescaled)
	beta_[N-1, :] = 1
	debug=False
	for n in range(N-2, -1, -1): # N-2 ~ 0
		# axis0: n, axis1: n+1
		bet = np.expand_dims(beta_[n+1, :], axis=0)
		p = np.expand_dims(p_x_given_z[n+1, :], axis=0)
		beta_[n, :] = (1/c[n+1])*np.sum(bet * p * A, axis=1) 
		if debug:
			print(bet)
			print(p)
			print(A)
			print(c[n+1])
			print(np.sum(bet * p * A, axis=1) )
			print("ho!")
			debug=False

	# compute results
	gamma = alpha_ * beta_
	xi = np.zeros((N-1, K, K))
	for n in range(1, N):
		xi[n-1,:,:] = (1/c[n])*np.expand_dims(alpha_[n-1,:],axis=1)*np.expand_dims(p_x_given_z[n,:],axis=0)*A*np.expand_dims(beta_[n,:],axis=0)
	avgLL = np.mean(np.log(c)) # log(P(X))/N

	return gamma, xi, avgLL

def main():
	x, z = generate_data()
	params = init_params(x)
	_, _, MU, SIGMA = params
	visualize_mixture(MU, SIGMA, x)
	eps = 10**(-7)
	oldLL = -10**5
	for _ in range(20):
		params, avgLL = em_step(params, x)
		if avgLL - oldLL < eps:
			break
		oldLL = avgLL
		print(avgLL)
	report_params(params)
	_, _, MU, SIGMA = params
	visualize_mixture(MU, SIGMA, x)

def report_params(params):
	pi, A, MU, SIGMA = params
	print("Transition probabilities:")
	for j in range(K):
		for k in range(K):
			print("%.03f  "%(A[j, k]), end="")
		print("")

def visualize_mixture(MU, SIGMA, x):	
	KEY_COLORS = 0.99*np.array([[1,0,0], [0,1,0], [0,0,1]])
	fig = plt.figure(0)
	ax = fig.add_subplot(111, aspect='equal')
	ax.scatter(x[:,0], x[:,1], marker='.')
	# make cluster ellipse
	for k in range(K):
		u, sig, vh = np.linalg.svd(SIGMA[k])
		angle = np.arctan2(u[1,0], u[0,0])
		r = 5
		e = Ellipse(MU[k], r*np.sqrt(sig[0]), r*np.sqrt(sig[1]), angle*180/np.pi)
		e.set_alpha(0.2)
		e.set_facecolor(KEY_COLORS[k])
		ax.add_artist(e)
	plt.show()


main()