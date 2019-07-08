import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 128
LATENT_DIM = 100
EPOCHS = 5

visual_path = 'data/visual/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data

transform_loc = (0.5, 0.5, 0.5)
transform_scale = (0.5, 0.5, 0.5)

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(transform_loc, transform_scale)
])

train_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

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

class Decoder(nn.Module):
	def __init__(self, input_dim):
		super(Decoder, self).__init__()

		# MLP decoder
		self.fc1 = nn.Linear(input_dim, 1000)
		self.fc2 = nn.Linear(1000, 28 * 28)

	def forward(self, z):
		a1 = F.relu(self.fc1(z))
		a2 = torch.tanh(self.fc2(a1))
		unflat = a2.view(-1, 1, 28, 28)
		return unflat

def test_autoencoder():

	encoder = Encoder(output_dim=LATENT_DIM)
	decoder = Decoder(input_dim=LATENT_DIM)
	encoder.to(device)
	decoder.to(device)

	criterion = nn.MSELoss()
	encoder_opt = optim.SGD(encoder.parameters(), lr=0.01)
	decoder_opt = optim.SGD(decoder.parameters(), lr=0.01)

	print_every = 100

	train = True
	if(train):
		print('Start training autoencoder')
		for epoch in range(EPOCHS):
			running_loss = 0.0
			for i, data in enumerate(train_loader, 0):

				encoder_opt.zero_grad()
				decoder_opt.zero_grad()

				x = data[0].to(device)

				z = encoder(x)
				xh = decoder(z)
				loss = criterion(x, xh)

				loss.backward()
				encoder_opt.step()
				decoder_opt.step()

				running_loss += loss.item()

				if(i % print_every == (print_every-1)):
					print('Epoch %d: iter %d: loss: %0.4f'%(epoch, i, running_loss/print_every))
					running_loss = 0.0

		print('Training complete')

	print('visualize inputs')
	with torch.no_grad():
		x, _ = iter(train_loader).next()
		x = x.to(device)
		xh = decoder(encoder(x))
		visualize_reconstructions(x, xh)

def visualize_reconstructions(x, xh):
	fig = plt.figure()
	N = 4
	for n in range(N):
		plt.subplot(N, 2, 2*n+1)
		show_tensor_img(x[n])
		plt.subplot(N, 2, 2*n+2)
		show_tensor_img(xh[n])
	plt.savefig(visual_path+'visualize_reconstructions.png')

def show_tensor_img(img):
	img = img.cpu().numpy().transpose((1, 2, 0))
	img = 0.5 * img + 0.5
	img = np.clip(img, 0, 1)
	ones = np.array([[[1, 1, 1]]])
	img = img * ones
	plt.imshow(img)
	plt.pause(0.001)


test_autoencoder()