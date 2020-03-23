# implmentation of early few-shot learning methods

import torchvision


def load_omniglot():
	dataset = torchvision.datasets.Omniglot(
		root="./data", download=True, transform=torchvision.transforms.ToTensor()
	)
	return dataset

def siamese_expr():
	pass

def matching_expr():
	pass

def prototypical_expr():
	pass

def main():
	print("Welcome back")
	dataset = load_omniglot()
	print(dataset)
	pass

main()