import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal



def generate_data():
	N_data = 100
	mu1 = torch.FloatTensor([5, 5]).view(2, 1)
	mu2 = torch.FloatTensor([-5, -5]).view(2, 1)
	X = torch.zeros(2, N_data)
	for n in range(N_data):
		if np.random.rand() < 0.5:
			z = mu1 + torch.randn(2, 1)
		else:
			z = mu2 + torch.randn(2, 1)
		mvn = dist_x_given_z(z)
		x = mvn.sample()
		X[:, n] = x
	return X

def dist_x_given_z(z):
	theta = torch.FloatTensor([[2., -1.], [1., -2.]])
	mean = torch.mm(theta, z).t()
	cov = 0.01*torch.eye(2)
	mvn = MultivariateNormal(mean, cov)	
	return mvn

def log_likelihood(z, x):
	log_l = dist_x_given_z(z).log_prob(x)
	return log_l

def main():
	X = generate_data()
	z = torch.FloatTensor([0, 0]).view(2, 1)
	z.requires_grad_(True)
	log_l = log_likelihood(z, X[:, 0])
	print(log_l)

main()