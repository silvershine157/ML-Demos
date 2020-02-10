# Decoupled Weight Decay Regularization

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test1():

	batch_size = 128

	transform = transforms.Compose([
		transforms.ToTensor()
	])
	trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
	                                        download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR100(root='./data', train=False,
	                                       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
	                                         shuffle=False, num_workers=2)
	classes = ('plane', 'car', 'bird', 'cat',
	           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	net = models.vgg11(pretrained=True)
	net.to(device)
	optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.01)
	#optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.01)
	#optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=0.0001)
	for epoch in range(20):
		train_loss = train_epoch(net, trainloader, optimizer)
		test_loss = test_epoch(net, testloader)
		print("Train loss: %g \t Test loss: %g"%(train_loss, test_loss))


def test_epoch(net, loader):
	running_n = 0
	running_loss = 0.0
	for batch_idx, batch in enumerate(loader):
		x, y = batch
		x = x.to(device)
		y = y.to(device)
		out = net(x)
		loss = nn.functional.cross_entropy(out, y, reduction='mean')
		loss.backward()
		B = x.size(0)
		running_n += B
		running_loss += B*loss.detach().item()
	loss = running_loss/running_n
	return loss

def train_epoch(net, loader, optimizer):
	running_n = 0
	running_loss = 0.0
	for batch_idx, batch in enumerate(loader):
		optimizer.zero_grad()
		x, y = batch
		x = x.to(device)
		y = y.to(device)
		out = net(x)
		loss = nn.functional.cross_entropy(out, y, reduction='mean')
		loss.backward()
		optimizer.step()
		B = x.size(0)
		running_n += B
		running_loss += B*loss.detach().item()
	loss = running_loss/running_n
	return loss

test1()