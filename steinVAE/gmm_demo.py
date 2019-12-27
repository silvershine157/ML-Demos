import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

mu1 = torch.FloatTensor([5, 5])
mu2 = torch.FloatTensor([-5, -5])
theta = torch.FloatTensor([[2., -1.], [1., -2.]])

def generate_data():
	N_data = 100
	X = torch.zeros(2, N_data)
	Z = torch.zeros(2, N_data)
	for n in range(N_data):
		if np.random.rand() < 0.5:
			z = mu1 + torch.randn(2)
		else:
			z = mu2 + torch.randn(2)
		Z[:, n] = z
	mean = torch.mm(theta, Z)
	X = mean + 0.1*torch.randn(2, N_data)
	return X

def log_prior(z):
	lp_1 = MultivariateNormal(mu1, torch.eye(2)).log_prob(z)
	lp_2 = MultivariateNormal(mu2, torch.eye(2)).log_prob(z)
	log_pri = torch.log((torch.exp(lp_1)+torch.exp(lp_2))/2)
	return log_pri

def log_likelihood(x, z):
	mean = torch.mm(theta, z.view(2, -1)).view(-1, 2)
	log_l = MultivariateNormal(mean, 0.01*torch.eye(2)).log_prob(z)
	return log_l

# unnormalized posterior
def log_joint(z, x): 
	log_j = log_likelihood(x, z) + log_prior(z)
	return log_j

def main():
	X = generate_data()
	z = torch.zeros(2)
	z.requires_grad_(True)
	log_j = log_joint(z, X[:, 0])
	print(log_j)

main()