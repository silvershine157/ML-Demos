import torch
import numpy as np
import matplotlib.pyplot as plt
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_gp(n, rbf_param=3.0):
    '''
    x: [n, 1]
    y: [n, 1]
    '''
    #x = 2.0*torch.randn(n, 1).to(device)
    x = torch.linspace(-3., 3., n).view(n, 1).to(device)
    # construct RBF kernel matrix
    D = x - x.t()
    cov = torch.exp(-D**2/rbf_param)+0.0001*torch.eye(n).to(device)
    mu = torch.zeros(n).to(device)
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
    y = 1.0*mvn.sample().view(n, 1).to(device)
    return x, y


def sample_obs(x_all, y_all, B, N):
    '''
    x_all: [n, x_dim]
    y_all: [n, y_dim]
    ---
    x_obs: [B, N, x_dim]
    y_obs: [B, N, y_dim]
    '''
    x_obs = torch.zeros(B, N, x_all.size(1), device=device)
    y_obs = torch.zeros(B, N, y_all.size(1), device=device)
    for b in range(B):
        p = torch.randperm(x_all.size(0))
        idx = p[:N]
        x_obs[b, :, :] = x_all[idx, :]
        y_obs[b, :, :] = y_all[idx, :]
    return x_obs, y_obs


def sample_gp_task(n, B):
	'''
	x_obs: [B, N, 1]
	y_obs: [B, N, 1]
	x_tar: [B, n, 1]
	y_tar: [B, n, 1]
	'''
	x_all, y_all = sample_gp(n, rbf_param=1.)
	N = np.random.randint(low=1, high=n)
	x_obs, y_obs = sample_obs(x_all, y_all, B, N)
	x_tar = x_all.unsqueeze(dim=0).expand(B, -1, -1)
	y_tar = y_all.unsqueeze(dim=0).expand(B, -1, -1)
	return x_obs, y_obs, x_tar, y_tar


def get_gaussian_params(out):
	'''
	out: [B, n, 2]
	---
	mean: [B, n, 1]
	var: [B, n, 1]
	'''
	mean = out[:, :, 0].unsqueeze(dim=2)
	#var = torch.exp(out[:, :, 1]).unsqueeze(dim=2)
	var = torch.log(torch.exp(out[:, :, 1])+1.0).unsqueeze(dim=2)
	return mean, var


def test():

	n = 30
	B = 32
	x_dim = 1
	y_dim = 1
	out_dim = 2
	r_dim = 128
	net = CNP(x_dim, y_dim, out_dim, r_dim).to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

	running_loss = 0.0
	running_cnt = 0

	for _ in range(100000):

		optimizer.zero_grad()
		x_obs, y_obs, x_tar, y_tar = sample_gp_task(n, B)
		out = net(x_obs, y_obs, x_tar)
		mean, var = get_gaussian_params(out)
		loss = NLLloss(y_tar, mean, var)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		running_cnt += 1
		if running_cnt == 1000:
			print(running_loss/running_cnt)
			running_loss = 0.0
			running_cnt = 0	

	torch.save(net, 'data/ckpt')

	with torch.no_grad():
		for _ in range(3):
			x_obs, y_obs, x_tar, y_tar = sample_gp_task(n, 1)
			out = net(x_obs, y_obs, x_tar)
			mean, var = get_gaussian_params(out)
			plot_result(x_obs, y_obs, x_tar, y_tar, mean, var)



def plot_result(x_obs, y_obs, x_tar, y_tar, mean, var):
	'''
	x_obs: [1, N, 1]
	y_obs: [1, N, 1]
	x_tar: [1, n, 1]
	y_tar: [1, n, 1]
	mean: [1, n, 1]
	var: [1, n, 1]
	'''
	x_obs = x_obs.squeeze().cpu().numpy()
	y_obs = y_obs.squeeze().cpu().numpy()
	x_tar = x_tar.squeeze().cpu().numpy()
	y_tar = y_tar.squeeze().cpu().numpy()
	mean = mean.squeeze().cpu().numpy()
	var = var.squeeze().cpu().numpy()
	plt.scatter(x_tar, y_tar, label='tar')
	plt.scatter(x_obs, y_obs, label='obs')
	#plt.scatter(x_tar, mean, label='mean')

	sort_idx = np.argsort(x_tar)
	x_sort = x_tar[sort_idx]
	mean_sort = mean[sort_idx]
	var_sort = var[sort_idx]
	std_sort = np.sqrt(var_sort)
	plt.plot(x_sort, mean_sort, label='pred mean', color='grey')
	plt.fill_between(x_sort, mean_sort-3*std_sort, mean_sort+3*std_sort, color='grey',alpha=0.3)

	plt.legend()
	plt.show()



test()