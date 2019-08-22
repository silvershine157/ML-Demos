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

class EmbeddingModule(nn.Module):
	def __init__(self):
		super(EmbeddingModule, self).__init__()
		self.feature_dim = 10
		self.conv = nn.Conv2d(1, self.feature_dim, kernel_size=3, padding=0)

	def forward(self, img):
		# img: B x 1 x 28 x 28
		# features: B x C x D x D
		features = self.conv(img)
		return features

class RelationModule(nn.Module):
	def __init__(self):
		super(RelationModule, self).__init__()
		self.sigmoid = nn.Sigmoid()

	def forward(self, combined):
		# combined: B x 2C x D x D
		# relation_scores: B
		relation_scores = self.sigmoid(combined.sum(keepdim=0))
		return relation_scores

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
			img = img.type(torch.FloatTensor)
			img = img/255
			imgs.append(img)
		random.shuffle(imgs)
		sample = imgs.pop()
		query = torch.cat(imgs, dim=0)
		return {"sample":sample, "query":query}


def combine_pairs(sample_features, query_features):

	_, C, D, _ = sample_features.size()

	# expand for pairing
	sample_features = sample_features.unsqueeze(dim=1).expand(-1, 95, -1, -1, -1) # 5 x 95 x C x D x D
	query_features = query_features.unsqueeze(dim=0).expand(5, -1, -1, -1, -1) # 5 x 95 x C x D x D
	
	combined = torch.cat([sample_features, query_features], dim=2) # 5 x 95 x 2C x D x D
	combined = combined.view(-1, 2*C, D, D) # 475 x 2C x D x D

	return combined, score_target


def test():
	# one pass
	meta_train_dirs, meta_val_dirs, meta_test_dirs = get_class_dirs()
	dataset = OmniglotOneshotDataset(meta_train_dirs)
	dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

	embed_net = EmbeddingModule()
	rel_net = RelationModule()

	for episode in range(EPISODES):

		# form episode
		ep_data = next(iter(dataloader))
		sample = ep_data['sample'] # 5 x 1 x 28 x 28
		query = ep_data['query'] # 5 x 19 x 28 x 28

		# flatten data
		sample = sample.view(-1, 1, 28, 28) # no change for one shot setting
		query = query.view(-1, 1, 28, 28)

		# do not want to recompute for every pair
		sample_features = embed_net(sample) # 5 x C x D x D
		query_features = embed_net(query) # 95 x C x D x D

		combined, score_target = combine_pairs(sample_features, query_features)

		print(combined.shape)

		break


def test2():
	pass

test()