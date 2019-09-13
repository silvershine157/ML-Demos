import os
import sys
import numpy as np

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
