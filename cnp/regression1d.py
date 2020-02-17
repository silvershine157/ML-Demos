import torch
import numpy as np
import matplotlib.pyplot as plt
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample_gp(n):
    '''
    x: [n, 1]
    y: [n, 1]
    '''
    x = 2.0*torch.randn(n, 1)
    #x = torch.linspace(-2., 2., n).view(n, 1)
    # construct RBF kernel matrix
    D = x - x.t()
    K = torch.exp(-D**2/10.0)+0.0001*torch.eye(n)
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n), K)
    y = 1.0*mvn.sample().view(n, 1)
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


def test1():
	n = 50
	x_dim = 1
	y_dim = 1
	out_dim = 2 # mean and var
	r_dim = 128  #128

	net = CNP(x_dim, y_dim, out_dim, r_dim).to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	for _ in range(10000):

		x_all, y_all = sample_gp(n)
		x_obs, y_obs = sample_obs(x_all, y_all, B=1, N=10)
		x_all = x_all.view(1, -1, 1).to(device)
		y_all = y_all.view(1, -1, 1).to(device)
		x_obs = x_obs.view(1, -1, 1).to(device)
		y_obs = y_obs.view(1, -1, 1).to(device)

		optimizer.zero_grad()
		out = net(x_obs, y_obs, x_all)
		mean = out[:, :, 0].unsqueeze(dim=2)
		loss = torch.mean((y_all - mean)**2)
		loss.backward()
		optimizer.step()
		print(loss.item())

	with torch.no_grad():
		for _ in range(10):
			x_all, y_all = sample_gp(n)
			x_obs, y_obs = sample_obs(x_all, y_all, B=1, N=10)
			x_all = x_all.view(1, -1, 1).to(device)
			y_all = y_all.view(1, -1, 1).to(device)
			x_obs = x_obs.view(1, -1, 1).to(device)
			y_obs = y_obs.view(1, -1, 1).to(device)
			y_out = net(x_obs, y_obs, x_all)[:, :, 0]
			plt.scatter(x_all.squeeze().cpu(), y_all.squeeze().cpu(), label='tar')
			plt.scatter(x_all.squeeze().cpu(), y_out.squeeze().cpu(), label='pred')
			plt.scatter(x_obs.squeeze().cpu(), y_obs.squeeze().cpu(), label='obs')
			plt.legend()
			plt.show()


test1()