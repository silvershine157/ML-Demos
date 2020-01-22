import os
import sys
import numpy as np
import torch
from torch.utils.data import random_split, Dataset, DataLoader

source_dir = 'data/source'

def download_data():
	download_links = [
		"http://www.openslr.org/resources/12/dev-clean.tar.gz"	
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
		os.system('tar -xzf '+os.path.join(source_dir, f)+' -C '+source_dir)

def preprocess_data():
	# read labels & construct vocabulary
	train_root = ""
	test_root = ""
	train_raw_label = scan_labels(train_root)
	test_raw_label = scan_labels(test_root)
	voc = make_voc(train_raw_label)
	# train/val split
	val_ratio=0.2
	n_val = int(val_ratio * len(train_raw_label))
	train_raw_label, val_raw_label = train_raw_label[n_val:], train_raw_label[:n_val]
	# convert to flat spectrogram, label (indices)
	train_specgram, train_label = make_flat_data(train_root, train_raw_label, voc)
	val_specgram, val_label = make_flat_data(val_root, val_raw_label, voc)
	test_specgram, test_label = make_flat_data(test_root, test_raw_label, voc)

	data = {}
	data["voc"] = voc # Voc object
	data["train_specgram"] = train_specgram # list of tensors
	data["train_label"] = train_label  # list of list of indices
	data["val_specgram"] = val_specgram
	data["val_label"] = val_label
	data["test_specgram"] = test_specgram
	data["test_label"] = test_label
	return data

def scan_labels(root):
	raw_label = None # list(speaker) of list(chapter) of string
	return raw_label

class Voc(object):
	def __init__(self):
		self.idx2char = {}
		self.char2idx = {}
	
def make_voc(raw_label):
	voc = None
	return voc

def make_flat_data(root, row_label, voc):
	specgram = None # list of tensors
	label = None # list of list of indices
	return specgram, label



# template code

class MnistDataset(Dataset):
	def __init__(self, imgs, labels):
		self.imgs = imgs
		self.labels = labels
	def __len__(self):
		return self.imgs.shape[0]
	def __getitem__(self, idx):
		img = torch.from_numpy(self.imgs[idx, :, :]).type(torch.FloatTensor)
		label = torch.from_numpy(self.labels[idx, :]).type(torch.FloatTensor)
		return {"image":img, "label":label}

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

