from sklearn.datasets import load_boston
import numpy as np
import torch
from torch.utils.data import random_split, Dataset, DataLoader

def prepare_data():
	boston = load_boston()
	X = boston.data
	Y = boston.target
	Y = np.reshape(Y, (-1, 1))
	train_X, train_Y, test_X, test_Y = shuffle_and_split(X, Y)
	train_X, train_Y, mu, sigma = rescale_data(train_X, train_Y)
	test_X, test_Y, _, _ = rescale_data(test_X, test_Y, mu, sigma)
	data = {}
	data["train_X"] = train_X
	data["train_Y"] = train_Y
	data["test_X"] = test_X
	data["test_Y"] = test_Y
	return data

def rescale_data(X, Y, mu=None, sigma=None):
	X_Y = np.concatenate((X, Y), axis=1)
	if mu is None or sigma is None:
		mu = np.mean(X_Y, axis=0)
		sigma = np.std(X_Y, axis=0)
	X_Y = (X_Y - mu)/(sigma + 1.0E-7)
	X = X_Y[:, :-1]
	Y = X_Y[:, -1].reshape(-1, 1)
	return X, Y, mu, sigma


def shuffle_and_split(X, Y):
	X_Y = np.concatenate((X, Y), axis=1)
	np.random.shuffle(X_Y)
	X = X_Y[:, :-1]
	Y = X_Y[:, -1].reshape(-1, 1)
	train_ratio = 0.8
	n = X.shape[0]
	n_train = int(train_ratio * n)
	train_X = X[:n_train, :]
	train_Y = Y[:n_train]
	test_X = X[n_train:, :]
	test_Y = Y[n_train:]
	return (train_X, train_Y, test_X, test_Y)

class BostonDataset(Dataset):
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
	def __len__(self):
		return self.X.shape[0]
	def __getitem__(self, idx):
		x = torch.from_numpy(self.X[idx, :]).type(torch.FloatTensor)
		y = torch.from_numpy(self.Y[idx]).type(torch.FloatTensor)
		return {"x":x, "y":y}

def get_dataloader(data, batch_size):
	
	# make train/val/test datasets
	train_ratio = 0.8
	n_train_full = data["train_X"].shape[0]
	n_train = int(train_ratio * n_train_full)
	n_val = n_train_full - n_train
	train_full_ds = BostonDataset(data["train_X"], data["train_Y"])
	train_ds, val_ds = random_split(train_full_ds, [n_train, n_val])
	test_ds = BostonDataset(data["test_X"], data["test_Y"])
	
	# make dataloaders
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

	return train_loader, val_loader, test_loader


def test():
	data = prepare_data()
	train_loader, val_loader, test_loader = get_dataloader(data, 4)

