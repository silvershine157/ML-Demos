import os
import sys
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader

source_dir = 'data/source'

def download_data():
	download_links = [
		"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
	]
	# ensure clean source data directory
	if not os.path.isdir(source_dir):
		os.makedirs(source_dir)
	else:
		os.system('rm -rf '+source_dir+'/*')
	# download source data
	for link in download_links:
		os.system('wget -P ' + source_dir + ' ' + link)
	# extract data
	for f in os.listdir(source_dir):
		os.system('gzip -d ' + os.path.join(source_dir, f))

def preprocess_data():
	# process source data
	data = {}
	data["train_images"] = read_mnist_images("train-images-idx3-ubyte")
	data["train_labels"] = read_mnist_labels("train-labels-idx1-ubyte")
	data["test_images"] = read_mnist_images("t10k-images-idx3-ubyte")
	data["test_labels"] = read_mnist_labels("t10k-labels-idx1-ubyte")
	return data

def read_mnist_images(fname):
	with open(os.path.join(source_dir, fname), 'rb') as f:
		magic = read_bytes(f, 4)
		if magic != 2051:
			return ValueError
		n_imgs = read_bytes(f, 4)
		rows = read_bytes(f, 4)
		cols = read_bytes(f, 4)
		dt = np.dtype(np.uint8).newbyteorder('>')
		imgs = np.fromfile(f, dtype=dt)
	imgs = np.reshape(imgs, (n_imgs, rows, cols))
	imgs = imgs/255.0 # normalize to 0~1
	return imgs

def read_mnist_labels(fname):
	with open(os.path.join(source_dir, fname), 'rb') as f:
		magic = read_bytes(f, 4)
		if magic != 2049:
			return ValueError
		n_labels = read_bytes(f, 4)
		dt = np.dtype(np.uint8).newbyteorder('>')
		labels = np.fromfile(f, dtype=dt)
	labels = np.reshape(labels, (n_labels))
	labels = np.eye(10)[labels] # one-hot encoding
	return labels

def read_bytes(f, num_bytes):
	b = f.read(num_bytes)
	return int.from_bytes(b, byteorder='big')

class MnistDataset(Dataset):
	def __init__(self, imgs, labels):
		self.imgs = imgs
		self.labels = labels
	def __len__(self):
		return self.imgs.shape[0]
	def __getitem(self, idx):
		return {"image":self.imgs[idx, :, :], "label":self.imgs[idx]}

def get_dataloader(data, batch_size):
	
	# make train/val/test datasets
	train_ratio = 0.8
	n_train_full = data["train_images"].shape[0]
	n_train = int(train_ratio * n_train_full)
	n_val = n_train_full - n_train
	train_full_ds = MnistDataset(data["train_images"], data["train_labels"])
	train_ds, val_ds = random_split(train_full_ds, [n_train, n_val])
	test_ds = MnistDataset(data["test_images"], data["test_labels"])
	
	# make dataloaders
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

	return train_loader, val_loader, test_loader

