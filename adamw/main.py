# Decoupled Weight Decay Regularization

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def test1():
	vgg16 = models.vgg11(pretrained=True)

def test2():
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
	                                        download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
	                                          shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR100(root='./data', train=False,
	                                       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4,
	                                         shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat',
	           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	

test2()