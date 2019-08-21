import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

BATCH_SIZE = 32
HIDDEN_DIM = 1000
LATENT_DIM = 30
EPOCHS = 3
learning_rate = 1.0

mini_data = False
mini_size = 1000

print_every = 100


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
test_data = torchvision.datasets.MNIST('data/mnist', train=False, download=True, transform=transform)

if(mini_data):
	print('use mini data')
	train_indices = list(range(mini_size))
	train_sampler = SubsetRandomSampler(train_indices)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
else:
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

class MLPEncoder(nn.Module):
	def __init__(self, output_dim):
		super(MLPEncoder, self).__init__()
		
		# MLP encoder
		self.fc1 = nn.Linear(28 * 28, HIDDEN_DIM)
		self.fc2 = nn.Linear(HIDDEN_DIM, output_dim)

	def forward(self, x):
		x_flat = x.view(-1, 28 * 28)
		a1 = F.relu(self.fc1(x_flat))
		a2 = torch.tanh(self.fc2(a1))
		return a2

class MLPDecoder(nn.Module):
	def __init__(self, input_dim):
		super(MLPDecoder, self).__init__()

		# MLP decoder
		self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
		self.fc2 = nn.Linear(HIDDEN_DIM, 28 * 28)

	def forward(self, z):
		a1 = F.relu(self.fc1(z))
		a2 = torch.tanh(self.fc2(a1))
		unflat = a2.view(-1, 1, 28, 28)
		return unflat

class Shallow(nn.Module):
	##The network struggles to learn identity transform even for the shallow case
	def __init__(self, hidden_dim):
		super(Shallow, self).__init__()
		self.fc1 = nn.Linear(28 * 28, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, 28 * 28)

	def forward(self, x):
		x_flat = x.view(-1, 28 * 28)
		a1 = torch.tanh(self.fc1(x_flat))
		a2 = torch.tanh(self.fc2(a1))
		unflat = a2.view(-1, 1, 28, 28)

		return unflat


def test_autoencoder():

	'''
	Working configuration:

	BATCH_SIZE = 16
	HIDDEN_DIM = 1000
	LATENT_DIM = 100
	EPOCHS = 2
	learning_rate = 0.3

	BATCH_SIZE = 16
	HIDDEN_DIM = 1000
	LATENT_DIM = 10
	EPOCHS = 2
	learning_rate = 0.3

	'''

	encoder = MLPEncoder(output_dim=LATENT_DIM)
	decoder = MLPDecoder(input_dim=LATENT_DIM)
	encoder.to(device)
	decoder.to(device)

	criterion = nn.MSELoss()
	encoder_opt = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_opt = optim.SGD(decoder.parameters(), lr=learning_rate)

	train = True
	if(train):
		print('Start training autoencoder')
		for epoch in range(EPOCHS):
			running_loss = 0.0
			total_loss = 0.0
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
				total_loss += loss.item()

				if(i % print_every == (print_every-1)):
					print('Epoch %d: iter %d: loss: %0.4f'%(epoch, i, running_loss/print_every))
					running_loss = 0.0

			print('Epcoh %d: total loss: %0.4f'%(epoch, total_loss))

		print('Training complete')

	print('visualize inputs')
	with torch.no_grad():
		x, _ = iter(train_loader).next()
		x = x.to(device)
		xh = decoder(encoder(x))
		visualize_reconstructions(x, xh)

	if (LATENT_DIM == 2):
		print('drawing grid')
		with torch.no_grad():
			grid_sz = 10
			#(G, G, 2)
			gridvals = np.linspace(0.0, 1.0, grid_sz)
			grid = []
			for x in gridvals:
				row = []
				for y in gridvals:
					row.append([x, y])
				grid.append(row)

			grid = np.array(grid)
			grid = torch.Tensor(grid)

			#(G * G, 2)
			z = grid.view(-1, 2)
			#print(z.shape)
			#print(z)

			z = z.to(device)
			xh = decoder(z)

			fig = plt.figure()
			for i in range(grid_sz):
				for j in range(grid_sz):
					idx = grid_sz*i + j
					plt.subplot(grid_sz, grid_sz, idx+1)
					show_tensor_img(xh[idx])
			plt.savefig(visual_path+'visualize_grid.png')



def test_shallow():
	shallow = Shallow(LATENT_DIM)
	shallow.to(device)
	criterion = nn.MSELoss()
	opt = optim.SGD(shallow.parameters(), lr=learning_rate)

	train = True
	if(train):
		print('Start training shallow autoencoder')
		for epoch in range(EPOCHS):
			running_loss = 0.0
			total_loss = 0.0
			for i, data in enumerate(train_loader, 0):

				opt.zero_grad()

				x = data[0].to(device)

				xh = shallow(x)
				loss = criterion(x, xh)

				loss.backward()
				opt.step()

				running_loss += loss.item()
				total_loss += loss.item()

				if(i % print_every == (print_every-1)):
					print('Epoch %d: iter %d: loss: %0.4f'%(epoch, i, running_loss/print_every))
					running_loss = 0.0

			print('Epcoh %d: total loss: %0.4f'%(epoch, total_loss))

		print('Training complete')

	print('visualize inputs')
	with torch.no_grad():
		x, _ = iter(test_loader).next()
		x = x.to(device)
		xh = shallow(x)
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
