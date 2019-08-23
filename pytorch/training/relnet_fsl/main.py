import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from skimage import io
import os
import random
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Meta train set
- Sample set
- Query set
Meta test set
- Support set
- Test set
'''

EPISODES = 1000
LEARNING_RATE = 0.001
VALIDATE_EVERY = 100
VALIDATE_EPS = 25
PRINT_EVERY = 50

'''
network architecture is identical to the author's code:
https://github.com/floodsung/LearningToCompare_FSL/blob/master/omniglot/omniglot_train_one_shot.py
'''
class EmbeddingModule(nn.Module):
	def __init__(self):
		super(EmbeddingModule, self).__init__()

		# define architecture
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=3, padding=0),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=0),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=0),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU()
		)
		self.layer4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=0),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
		)

	def forward(self, img):
		# img: B x 1 x 28 x 28
		# out: B x C x D x D
		out = self.layer1(img)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)		
		return out

class RelationModule(nn.Module):
	def __init__(self):
		super(RelationModule, self).__init__()
		
		input_size = 64
		hidden_size = 8
		# define architecture
		self.layer1 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.out_layer = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1),
			nn.Sigmoid()
		)

	def forward(self, combined):
		# combined: B x 2C x D x D
		# out: B
		out = self.layer1(combined)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.out_layer(out)
		return out

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

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		if m.bias is not None:
			m.bias.data.zero_()
		elif classname.find('BatchNorm') != -1:
			m.weight.data.fill_(1)
			m.bias.data.zero_()
		elif classname.find('Linear') != -1:
			n = m.weight.size(1)
			m.weight.data.normal_(0, 0.01)
			m.bias.data = torch.ones(m.bias.data.size())



def combine_pairs(sample_features, query_features):

	_, C, D, _ = sample_features.size()
	
	# generate labels
	sample_classes = torch.arange(5) # 5
	query_classes = sample_classes.unsqueeze(dim=1).expand(-1, 19) # 5 x 19

	# expand dimensions
	sample_classes = sample_classes.unsqueeze(dim=1).expand(-1, 95) # 5 x 95	
	query_classes = query_classes.contiguous().view(1, -1).expand(5, -1) # 5 x 95
	
	# generate target
	target = (sample_classes == query_classes).type(torch.FloatTensor)
	target = target.view(-1, 1) # 475 x 1

	# expand for pairing
	sample_features = sample_features.unsqueeze(dim=1).expand(-1, 95, -1, -1, -1) # 5 x 95 x C x D x D
	query_features = query_features.unsqueeze(dim=0).expand(5, -1, -1, -1, -1) # 5 x 95 x C x D x D
	
	# concat in depth
	combined = torch.cat([sample_features, query_features], dim=2) # 5 x 95 x 2C x D x D
	combined = combined.view(-1, 2*C, D, D) # 475 x 2C x D x D
	
	return combined, target


def test():

	# setup data
	meta_train_dirs, meta_val_dirs, meta_test_dirs = get_class_dirs()
	metatrain_dataset = OmniglotOneshotDataset(meta_train_dirs)
	metatrain_loader = DataLoader(metatrain_dataset, batch_size=5, shuffle=True)
	metaval_dataset = OmniglotOneshotDataset(meta_val_dirs)
	metaval_loader = DataLoader(metaval_dataset, batch_size=5, shuffle=True)
	
	# construct model
	print('Constructing model . . .')
	embed_net = EmbeddingModule()
	rel_net = RelationModule()
	criterion = nn.MSELoss()

	# setup training
	embed_opt = torch.optim.Adam(embed_net.parameters(), lr=LEARNING_RATE)
	rel_opt = torch.optim.Adam(rel_net.parameters(), lr=LEARNING_RATE)
	embed_scheduler = StepLR(embed_opt, step_size=100000, gamma=0.5)
	rel_scheduler = StepLR(rel_opt, step_size=100000, gamma=0.5)

	embed_net.apply(weights_init)
	rel_net.apply(weights_init)
	embed_net.to(device)
	rel_net.to(device)

	# training
	print('Training . . .')
	running_loss = 0.0
	for episode in range(1, EPISODES+1):
		
		embed_scheduler.step(episode)
		rel_scheduler.step(episode)

		# form episode
		ep_data = next(iter(metatrain_loader))
		sample = ep_data['sample'].to(device) # 5 x 1 x 28 x 28
		query = ep_data['query'].to(device) # 5 x 19 x 28 x 28
		query = query.view(-1, 1, 28, 28) # flattening, 95 x 1 x 28 x 28

		# forward pass
		sample_features = embed_net(sample) # 5 x C x D x D (avoid redundant computation)
		query_features = embed_net(query) # 95 x C x D x D
		combined, score_target = combine_pairs(sample_features, query_features)
		score_target = score_target.to(device)
		score_pred = rel_net(combined)
		loss = criterion(score_pred, score_target)

		# backward pass & update
		embed_net.zero_grad()
		rel_net.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(embed_net.parameters(), 0.5)
		nn.utils.clip_grad_norm_(rel_net.parameters(), 0.5)
		embed_opt.step()
		rel_opt.step()

		running_loss += loss.item()
		if (episode % PRINT_EVERY == 0):
			print('episode: ', episode, 'loss: ', running_loss/PRINT_EVERY)
			running_loss = 0.0
		
		if (episode % VALIDATE_EVERY == 0):
			correct = 0
			for _ in range(VALIDATE_EPS):
				ep_data = next(iter(metaval_loader))
				sample = ep_data['sample'].to(device)
				query = ep_data['query'].to(device)
				query = query.view(-1, 1, 28, 28)
				sample_features = embed_net(sample)
				query_features = embed_net(query)
				combined, score_target = combine_pairs(sample_features, query_features)
				score_target = score_target.to(device)
				# combined: 475 x 2C x D x D
				# target: 475
				score_pred = rel_net(combined) # 475
				target_class = torch.argmax(score_target.view(5, 95), dim=0)
				pred_class = torch.argmax(score_pred.view(5, 95), dim=0)
				equals = (target_class == pred_class)
				correct += torch.sum(equals).item()

			accuracy = correct/95/VALIDATE_EPS
			print('validation accuracy: ', accuracy)


# TODO: validation (preferably modularized), monitoring log, good initlialization, better target, learning rate
# schedulig


test()
