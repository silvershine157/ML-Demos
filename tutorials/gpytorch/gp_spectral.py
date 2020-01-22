import math
import torch
import gpytorch
from matplotlib import pyplot as plt

train_x = torch.linspace(0, 1, 15)
train_y = torch.sin(train_x * (2 * math.pi))

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
		self.covar_module.initialize_from_data(train_x, train_y)

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGPModel(train_x, train_y, likelihood)

# train: find optimal model hyperparameters
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
for i in range(training_iter):
	optimizer.zero_grad()
	output = model(train_x)
	loss = -mll(output, train_y)
	loss.backward()
	print('Iter %d/%d - Loss: %.3f'%(i+1, training_iter, loss.item()))
	optimizer.step()

# test
test_x = torch.linspace(0, 5, 51)
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
	observed_pred = likelihood(model(test_x))
	f, ax = plt.subplots(1, 1, figsize=(4,3))
	lower, upper = observed_pred.confidence_region()
	ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
	ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
	ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
	ax.set_ylim([-3, 3])
	ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.show()