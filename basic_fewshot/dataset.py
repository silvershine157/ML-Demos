import torchvision
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset


def load_omniglot():
	dataset = torchvision.datasets.Omniglot(
		root="./data", download=True, transform=torchvision.transforms.ToTensor()
	)
	loader = DataLoader(dataset, batch_size=20)
	per_class = []
	for i, batch in enumerate(loader):
		x, y = batch
		per_class.append(x)
	full_data = torch.stack(per_class) # [964, 20, 1, 105, 105]
	test_ratio = 0.2
	train_n = int((1.0-test_ratio)*full_data.size(0))
	train_data = full_data[:train_n] # [train_n, 20, 1, 105, 105]
	test_data = full_data[train_n:]
	return train_data, test_data


def get_siamese_loader(data, n_pairs, batch_size):
	'''
	data: [n_classes, 20, 1, 105, 105]
	---
	loader batch: ([B, 2, 1, 105, 105], [B])
	'''
	n_classes = data.size(0)
	pairs = torch.zeros((n_pairs, 2, 1, 105, 105))
	labels = torch.zeros((n_pairs))
	for i in range(n_pairs):
		if i%2==0:
			label = 1.0 # same class
			c_idx0 = random.randint(0, n_classes-1)
			c_idx1 = c_idx0
		else:
			label = 0.0 # different class
			c_idx0 = random.randint(0, n_classes-1)
			c_idx1 = (c_idx0+random.randint(1, n_classes-1))%n_classes
		pairs[i, 0] = data[c_idx0, random.randint(0, 19)]
		pairs[i, 1] = data[c_idx1, random.randint(0, 19)]
		labels[i] = label
	dataset = TensorDataset(pairs, labels)
	loader = DataLoader(dataset, batch_size=batch_size)
	return loader


class EpisodeDataset(Dataset):
	def __init__(self, data, n_episodes, C, K):
		self.data = data
		self.n_episodes = n_episodes
		self.C = C
		self.K = K

	def __len__(self):
		return self.n_episodes

	def __getitem__(self, idx):
		'''
		{
			"support": [C, K, 1, 105, 105]
			"query": [Q, 1, 105, 105]
			"label": [Q]
		}
		'''
		n_classes = self.data.size(0)
		n_examples = self.data.size(1)
		classes = np.random.choice(np.arange(n_classes), self.C, replace=False)
		ep_data = self.data[classes, :, :, :, :] # [C, n_examples, 1, 105, 105]
		perm = torch.randperm(n_examples)
		support_idx = perm[:self.K]
		query_idx = perm[self.K:]
		support = ep_data[:, support_idx, :, :, :]
		query = ep_data[:, query_idx, :, :, :].view((-1, 1, 105, 105))
		label = torch.arange(self.C).unsqueeze(1).repeat(1, n_examples-self.K).view(-1)
		item = {"support": support, "query": query, "label": label}
		return item


def get_episode_loader(data, C, K):
	'''
	data: [n_classes, 20, 1, 105, 105]
	---
	loader
	'''
	dataset = EpisodeDataset(data, 100, C, K)
	loader = DataLoader(dataset, batch_size=1)
	return loader