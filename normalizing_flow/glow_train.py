import torch
import torchvision

from glow import *

try_gpu = False
device = torch.device("cuda" if (try_gpu and torch.cuda.is_available()) else "cpu")

def test1():

	# prepare data
	batch_size = 256
	S = 32
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize((S, S)),
		torchvision.transforms.ToTensor()
	])
	#ds = torchvision.datasets.MNIST('./data/', download=True, transform=transform)
	ds = torchvision.datasets.CelebA('./data/', download=True, transform=transform)
	loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

	# prepare model
	K = 4
	L = 3
	C = 3
	glow = Glow(K, L, C)
	glow.to(device)

	# train model
	optimizer = torch.optim.Adam(glow.parameters())
	for batch in loader:
		x, y = batch
		x = x.to(device)
		optimizer.zero_grad()
		LL = glow.log_likelihood(x)
		loss = -LL.mean()
		loss.backward()
		optimizer.step()
		print(loss.item())


def test2():
	B = 3
	S = 8
	C = 3
	L = 1
	K = 2
	flow = Glow(K, L, C)
	x = torch.randn((B, C, S, S)) # dummy input
	z, _ = flow.forward_flow(x)
	x_r = flow.inverse_flow(z)
	print(torch.mean(torch.abs(x - x_r)))



test2()