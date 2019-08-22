import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
import os
import random
import numpy as np


'''
Meta train set
- Sample set
- Query set
Meta test set
- Support set
- Test set
'''

EPISODES = 3

class RelationNet(nn.Module):
	'''A dummy implementation of relation net'''
	def __init__(self):
		super(RelationNet, self).__init__()
		self.sigmoid = nn.Sigmoid()

	def forward(img1, img2):
		return self.sigmoid(torch.mean(img1+img2))


def get_class_dirs():

	# read all 1623 character class dir in omniglot
	data_dir = './data/omniglot_resized/'
	all_classes = []
	for family in os.listdir(data_dir):
		family_dir = os.path.join(data_dir, family)
		if not os.path.isdir(family_dir):
			continue
		for class_name in os.listdir(family_dir):
			class_dir = os.path.join(family_dir, class_name)
			if not os.path.isdir(class_dir):
				continue
			all_classes.append(class_dir)

	# split dataset
	num_train = 1200
	random.seed(1)
	random.shuffle(all_classes)
	meta_train_dirs = all_classes[:num_train] # 1200
	meta_val_dirs = all_classes[num_train:] # 423
	meta_test_dirs = meta_val_dirs # seems like the author's code is doing this

	return meta_train_dirs, meta_val_dirs, meta_test_dirs


class OmniglotOneshotDataset(Dataset):
	def __init__(self, dir_list):
		self.dir_list = dir_list

	def __len__(self):
		return len(self.dir_list)

	def __getitem__(self, idx):
		img_dir = self.dir_list[idx]
		imgs = []
		for img_name in os.listdir(img_dir):
			img = io.imread(os.path.join(img_dir, img_name))
			img = torch.from_numpy(img)
			img = img.unsqueeze(dim=0)
			img = img/255
			imgs.append(img)
		random.shuffle(imgs)
		sample = imgs.pop()
		query = torch.cat(imgs, dim=0)
		return {"sample":sample, "query":query}

def test():
	# one pass
	meta_train_dirs, meta_val_dirs, meta_test_dirs = get_class_dirs()
	net = RelationNet()
	dataset = OmniglotOneshotDataset(meta_train_dirs)

	dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

	for episode in range(EPISODES):

		ep_data = next(iter(dataloader))
		sample = ep_data['sample']
		query = ep_data['query']


		

def test2():
	pass

test()