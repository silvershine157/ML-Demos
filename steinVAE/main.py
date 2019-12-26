import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
	def __init__(self, latent_dim):
		super(Encoder, self).__init__()
		self.latent_dim = latent_dim
		self.layers = nn.Sequential(
			nn.Linear(28*28, 200),
			nn.Tanh(),
			nn.Linear(200, latent_dim*2)
		)
	def forward(self, x):
		B = x.size(0)
		out = self.layers(x.view(B, -1))
		mu, pre_sigma = out[:, :self.latent_dim], out[:, self.latent_dim:]
		sigmoid = nn.Sigmoid()
		sigma = sigmoid(pre_sigma)
		return mu, sigma

class Decoder(nn.Module):
	def __init__(self, latent_dim):
		super(Decoder, self).__init__()
		self.latent_dim = latent_dim
		self.layers = nn.Sequential(
			nn.Linear(latent_dim, 200),
			nn.Tanh(),
			nn.Linear(200, 28*28),
			nn.Sigmoid()
		)
	def forward(self, z):
		B = z.size(0)
		out = self.layers(z)
		x_r = out.view(B, 1, 28, 28)
		return x_r

def get_mnist_loaders(batch_size):
	mnist_transform = transforms.Compose([
	    transforms.ToTensor(), 
	    #transforms.Normalize((0.0,), (1.0,)),
	    lambda x: x > 0.3,
	    lambda x: x.float()
	])
	download_root = './data/MNIST_DATASET'
	train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
	valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
	test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
	return train_loader, valid_loader, test_loader

#print(x.shape)
#plt.imshow(x[0, 0, :, :].numpy())
#plt.show()

def D_KL(mu, sigma):
	out = -torch.mean(1 + torch.log(sigma**2) - mu**2 - sigma**2)
	return out

def main():

	latent_dim=50
	batch_size=32
	lr = 0.001
	train_loader, valid_loader, test_loader = get_mnist_loaders(batch_size)
	enc = Encoder(latent_dim).to(device)
	dec = Decoder(latent_dim).to(device)
	enc_optim = optim.Adam(enc.parameters(), lr=lr)
	dec_optim = optim.Adam(dec.parameters(), lr=lr)

	for epoch in range(10):
		running_loss = 0.0
		running_n = 0
		for x, y in train_loader:
			enc_optim.zero_grad()
			dec_optim.zero_grad()
			x = x.to(device)
			B = x.size(0)
			mu, sigma = enc(x)
			noise = torch.randn(B, latent_dim).to(device)
			z = mu + sigma * noise
			x_r = dec(z)
			LL_0 = torch.mean(torch.log(1-x_r[x < 0.5]))
			LL_1 = torch.mean(torch.log(x_r[x > 0.5]))
			LL = (LL_0+LL_1)/2
			loss = -(LL - D_KL(mu, sigma))
			loss.backward()
			enc_optim.step()
			dec_optim.step()
			running_n += B
			running_loss += B*loss.item()
		print(running_loss/running_n)

	enc.cpu()
	dec.cpu()
	for x, y in valid_loader:
		plt.imshow(x[0, 0, :, :].numpy())
		plt.show()	
		mu, sigma = enc(x)
		x_r = dec(mu).detach()
		plt.imshow(x_r[0, 0, :, :].numpy())
		plt.show()		

main()