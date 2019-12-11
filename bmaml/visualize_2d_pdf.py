import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import multivariate_normal

def test():
	n_grid=100
	x = np.linspace(-1, 1, n_grid)
	y = np.linspace(-1, 1, n_grid)
	xx, yy = np.meshgrid(x, y)
	xy = np.stack([xx, yy], axis=2)
	z = gmm_ex(xy)
	plt.imshow(z, interpolation='bilinear')
	plt.show()

def gmm_ex(xy):
	# xy: [n_x, n_y, 2]
	# z: [n_x, n_y]
	MU = [
		np.array([0.5, 0.5]),
		np.array([-0.5, -0.5])
	]
	SIGMA = [
		0.1*np.eye(2),
		0.4*np.eye(2)
	]
	PI = [
		0.3,
		0.7
	]
	n_x, n_y, _ = xy.shape
	z = np.zeros((n_x, n_y))
	for pi, mu, sigma in zip(PI, MU, SIGMA):
		z += pi*gaussian(xy, mu, sigma)
	return z

def gaussian(xy, mu, sigma):
	z = multivariate_normal.pdf(xy, mu, sigma)
	return z



test()