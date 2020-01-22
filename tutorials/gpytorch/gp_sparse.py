import math
import torch
import gpytorch
from matplotlib import pyplot as plt



## Load dataset

import urllib.request
import os
from scipy.io import loadmat
from math import floor

if not os.path.isfile('data/elevators.mat'):
	print('Downloading UCI dataset: elevators')
	urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', 'data/elevators.mat')
	print('Download complete')
print('Opening dataset')

data = torch.Tensor(loadmat('data/elevators.mat')['data'])
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()
test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
	train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()



## Define SPGR model

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

class GPRegressionModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = ConstantMean()
		self.base_covar_module = ScaleKernel(RBFKernel())
		self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500, :], likelihood=likelihood)

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)
if torch.cuda.is_available():
	model = model.cuda()
	likelihood = likelihood.cuda()


## Training
model.train()
likelihood.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 50
for i in range(training_iterations):
	optimizer.zero_grad()
	output = model(train_x)
	loss = -mll(output, train_y)
	loss.backward()
	print('Iter %d/%d - Loss: %.3f'%(i+1, training_iterations, loss.item()))
	optimizer.step()
	torch.cuda.empty_cache()



## Testing
model.eval()
likelihood.eval()
with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
	with gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
		preds = model(test_x)
print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))