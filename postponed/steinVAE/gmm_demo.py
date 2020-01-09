import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

mu1 = torch.FloatTensor([5, 5])
mu2 = torch.FloatTensor([-5, -5])
theta = torch.FloatTensor([[2., -1.], [1., -2.]])

def generate_data(N_data):
	X = torch.zeros(2, N_data)
	Z = torch.zeros(2, N_data)
	for n in range(N_data):
		Z[:, n] = sample_z()
	mean = torch.mm(theta, Z)
	X = mean + 0.1*torch.randn(2, N_data)
	return X

def sample_z():
	if np.random.rand() < 0.5:
		z = mu1 + torch.randn(2)
	else:
		z = mu2 + torch.randn(2)
	return z	

def log_prior(z):
	lp_1 = MultivariateNormal(mu1, torch.eye(2)).log_prob(z)
	lp_2 = MultivariateNormal(mu2, torch.eye(2)).log_prob(z)
	log_pri = torch.log((torch.exp(lp_1)+torch.exp(lp_2))/2)
	return log_pri

def log_likelihood(x, z):
	mean = torch.mm(theta, z.view(2, -1)).view(-1, 2)
	log_l = MultivariateNormal(mean, 0.01*torch.eye(2)).log_prob(x)
	return log_l

# unnormalized posterior
def log_joint(z, x): 
	log_j = log_likelihood(x, z) + log_prior(z)
	return log_j

def log_joint_analytic(z, x):
	sigma = 0.1
	Sigma = torch.matmul(theta.t(), theta)/(sigma**2) + torch.eye(2)
	SigmaInv = Sigma.inverse()
	mu1hat = torch.matmul(SigmaInv,torch.matmul(x, theta)/(sigma**2)-mu1)
	mu2hat = torch.matmul(SigmaInv,torch.matmul(x, theta)/(sigma**2)-mu2)
	quad1 = torch.matmul(mu1hat, torch.matmul(Sigma, mu1hat))
	quad2 = torch.matmul(mu2hat, torch.matmul(Sigma, mu2hat))
	p1 = torch.sum(x**2)/(sigma**2) + torch.sum(mu1**2) - quad1
	p2 = torch.sum(x**2)/(sigma**2) + torch.sum(mu2**2) - quad2
	phat = 1/(1+torch.exp(p2-p1))
	lp1 = MultivariateNormal(mu1hat, Sigma).log_prob(z)
	lp2 = MultivariateNormal(mu2hat, Sigma).log_prob(z)
	res = phat*torch.exp(lp1) + (1-phat)*torch.exp(lp2)
	return torch.log(res)

def posterior_visualization():
	X = generate_data(1)
	x = X[:, 0]
	x = torch.zeros(2)
	n_grid = 50
	grid_size = 7.
	z_grid = np.linspace(-grid_size, grid_size, n_grid)
	log_j_values = torch.zeros((n_grid, n_grid))
	for z1_i, z1 in enumerate(z_grid):
		for z2_i, z2 in enumerate(z_grid):
			z = torch.Tensor([z1, z2])
			val = log_joint_analytic(z, x)
			log_j_values[z1_i, z2_i] = val
	j_values = torch.exp(log_j_values)/100
	img = j_values.numpy()
	plt.imshow(img)
	plt.show()

def test_data():
	X = generate_data(100)
	z = torch.zeros(2)
	z.requires_grad_(True)
	log_j = log_joint(z, X[:, 0])
	print(log_j)

def test_analytic():
	x = torch.zeros(2)
	z = torch.zeros(2)
	log_j = log_joint_analytic(z, x)

posterior_visualization()