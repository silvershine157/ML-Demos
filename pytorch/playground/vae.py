import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)

class Encoder(nn.Module):
	def __init__(self, output_dim):
		super(Encoder, self).__init__()
		
		# MLP encoder
		self.fc1 = nn.Linear(28 * 28, 1000)
		self.fc2 = nn.Linear(1000, output_dim)

	def forward(self, x):
		x_flat = x.view(-1, 28 * 28)
		a1 = F.relu(self.fc1(x_flat))
		a2 = self.fc2(a1)
		return a2


def test_autoencoder():

	latent_dim = 2

	encoder = Encoder(latent_dim)
	train_iter = iter(train_loader)

	x, _ = train_iter.next()
	z = encoder(x)

	print(z.size())
	print(z)


test_autoencoder()