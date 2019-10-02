import numpy as np
import matplotlib.pyplot as plt

## model dimensions
K = 3 # number of latent states
N = 10 # number of data points
D = 2 # dimension of a data point

def generate_data():
	## true model parameters
	# p(z_0): initial distribution
	PI = np.array([0.2, 0.5, 0.3])
	# p(z_n | z_{n-1}): transition matrix
	A = np.array([
		[0.2, 0.3, 0.5],
		[0.5, 0.2, 0.3],
		[0.3, 0.5, 0.2]
	])
	# p(x_n | z_n): D-dim gaussian
	PHI_MU = np.array([
		[0.0, 0.0],
		[1.0, 1.0],
		[-1.0, 1.0]
	])
	PHI_SIGMA = np.array([
		[[0.5, 0.0], [0.0, 0.5]],
		[[0.5, 0.0], [0.0, 0.5]],
		[[0.5, 0.0], [0.0, 0.5]]
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
	if z:
		# color code latent variable
		pass
	else:
		# no color
		pass


def main():
	x, z = generate_data()
	visualize_data(x, z)

main()