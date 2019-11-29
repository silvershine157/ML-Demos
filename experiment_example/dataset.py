from sklearn.datasets import load_boston
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader

def prepare_data():
	boston = load_boston()
	X = boston.data
	Y = boston.target
	train_X, train_Y, test_X, test_Y = split_train_test(X, Y)
	data = {}
	data["train_X"] = train_X
	data["train_Y"] = train_Y
	data["test_X"] = test_X
	data["test_Y"] = test_Y
	return data

def split_train_test(X, Y):
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

test()