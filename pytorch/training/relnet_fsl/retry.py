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


CLASS_IN_EP = 3
QUERY_SIZE = 1
EPISODES = 1000
PRINT_EVERY = 200
LEARNING_RATE = 0.01
FEATURE_DIM = 1

class SimpleEmbeddingModule(nn.Module):
	def __init__(self):
		super(SimpleEmbeddingModule, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, FEATURE_DIM, kernel_size=3, padding=0),
			nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

	def forward(self, img):
		out = self.layer1(img)
		return out

class EmbeddingModule(nn.Module):
	def __init__(self):
		super(EmbeddingModule, self).__init__()

		# define architecture
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, FEATURE_DIM, kernel_size=3, padding=0),
			nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(FEATURE_DIM, FEATURE_DIM, kernel_size=3, padding=0),
			nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(FEATURE_DIM, FEATURE_DIM, kernel_size=3, padding=0),
			nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
			nn.ReLU()
		)
		self.layer4 = nn.Sequential(
			nn.Conv2d(FEATURE_DIM, FEATURE_DIM, kernel_size=3, padding=0),
			nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
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

class SimpleRelationModule(nn.Module):
	def __init__(self):
		super(SimpleRelationModule, self).__init__()
		hidden_dim = 8
		self.layer = nn.Sequential(
			nn.Linear(2*FEATURE_DIM, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)

	def forward(self, combined):
		out = combined.view(combined.size(0), -1)
		out = self.layer(out)
		return out

class RelationModule(nn.Module):
	def __init__(self):
		super(RelationModule, self).__init__()
		
		input_size = FEATURE_DIM
		hidden_size = 8
		# define architecture
		self.layer1 = nn.Sequential(
			nn.Conv2d(2*FEATURE_DIM, FEATURE_DIM, kernel_size=3, padding=1),
			nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(FEATURE_DIM, FEATURE_DIM, kernel_size=3, padding=1),
			nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),
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
		imgs = imgs[:QUERY_SIZE]
		query = torch.cat(imgs, dim=0)
		return {"sample":sample, "query":query}


def combine_pairs(sample_features, query_features):

	_, C, D, _ = sample_features.size()
	
	# generate labels
	sample_classes = torch.arange(CLASS_IN_EP) # 5
	query_classes = sample_classes.unsqueeze(dim=1).expand(-1, QUERY_SIZE) # 5 x 19

	# expand dimensions
	sample_classes = sample_classes.unsqueeze(dim=1).expand(-1, QUERY_SIZE*CLASS_IN_EP) # 5 x 95	
	query_classes = query_classes.contiguous().view(1, -1).expand(CLASS_IN_EP, -1) # 5 x 95
	
	# generate target
	target = (sample_classes == query_classes).type(torch.FloatTensor)
	target = target.view(-1, 1) # 475 x 1

	# expand for pairing
	sample_features = sample_features.unsqueeze(dim=1).expand(-1, QUERY_SIZE*CLASS_IN_EP, -1, -1, -1) # 5 x 95 x C x D x D
	query_features = query_features.unsqueeze(dim=0).expand(CLASS_IN_EP, -1, -1, -1, -1) # 5 x 95 x C x D x D
	
	# concat in depth
	combined = torch.cat([sample_features, query_features], dim=2) # 5 x 95 x 2C x D x D
	combined = combined.view(-1, 2*C, D, D) # 475 x 2C x D x D
	
	return combined, target



def test1():
	# fixed episode
	dirs, _, _ = get_class_dirs()
	dataset = OmniglotOneshotDataset(dirs)
	loader = DataLoader(dataset, batch_size=CLASS_IN_EP, shuffle=True)
	ep_data = next(iter(loader))
	sample = ep_data['sample'] # CLASS_IN_EP x 1 x 28 x 28
	query = ep_data['query'] # CLASS_IN_EP x QUERY_SIZE x 28 x 28
	
	# use dummy data
	sample = torch.arange(CLASS_IN_EP).type(torch.FloatTensor)
	sample = sample.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
	sample = sample.expand(-1, -1, 28, 28).contiguous()
	query = sample.expand(-1, QUERY_SIZE, -1, -1).contiguous()
	query = query.view(-1, 1, 28, 28)

	combined_raw, target = combine_pairs(sample, query)
	print(combined_raw.shape)
	print(combined_raw[:, :, 1, 1])
	print(target.shape)
	print(target)

	embed_net = EmbeddingModule()
	rel_net = RelationModule()
	# use simple network
	#embed_net = SimpleEmbeddingModule()
	print(query.shape)
	query_feature = embed_net(query)
	print(query_feature.shape)

	criterion = nn.MSELoss()

	# setup training
	embed_opt = torch.optim.Adam(embed_net.parameters(), lr=LEARNING_RATE)
	rel_opt = torch.optim.Adam(rel_net.parameters(), lr=LEARNING_RATE)
	embed_scheduler = StepLR(embed_opt, step_size=100000, gamma=0.5)
	rel_scheduler = StepLR(rel_opt, step_size=100000, gamma=0.5)

	running_loss = 0.0
	for episode in range(1, EPISODES+1):
		sample_features = embed_net(sample)
		query_features = embed_net(query)
		combined, score_target = combine_pairs(sample_features, query_features)
		score_pred = rel_net(combined)
		loss = criterion(score_pred, score_target)
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
			'''
			print('target')
			print(score_target.view(-1).numpy())
			print('pred')
			print(score_pred.detach().view(-1).numpy())
			'''

	print(score_pred.detach().numpy())


def test2():
	# only train relation module

	# use dummy data
	sample = torch.arange(CLASS_IN_EP).type(torch.FloatTensor)
	sample = sample.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
	sample = sample.expand(-1, -1, 28, 28).contiguous()
	query = sample.expand(-1, QUERY_SIZE, -1, -1).contiguous()
	query = query.view(-1, 1, 28, 28)
	sample_features = (sample.expand(-1, FEATURE_DIM, -1, -1))[:, :, 0:1, 0:1] - 1
	query_features = (query.expand(-1, FEATURE_DIM, -1, -1))[:, :, 0:1, 0:1] - 1
	print(sample_features.shape)
	print(query_features.shape)	
	combined, score_target = combine_pairs(sample_features, query_features)
	print(combined.shape)

	score_target = 0.8*score_target

	# rel_net = RelationModule()
	# use simple network
	rel_net = SimpleRelationModule()

	criterion = nn.MSELoss()

	# setup training
	rel_opt = torch.optim.Adam(rel_net.parameters(), lr=LEARNING_RATE)
	
	running_loss = 0.0
	for episode in range(1, EPISODES+1):
		
		score_pred = rel_net(combined)
		loss = criterion(score_pred, score_target)
		rel_net.zero_grad()
		loss.backward()
		rel_opt.step()
		
		running_loss += loss.item()
		if (episode % PRINT_EVERY == 0):
			print('episode: ', episode, 'loss: ', running_loss/PRINT_EVERY)
			running_loss = 0.0

	print(combined[:, 0, 0, 0].view(CLASS_IN_EP, -1).numpy())
	print(combined[:, FEATURE_DIM, 0, 0].view(CLASS_IN_EP, -1).numpy())
	print(score_target.view(CLASS_IN_EP, -1).numpy())
	print(score_pred.view(CLASS_IN_EP, -1).detach().numpy())
	# simple regression not working???


class VerySimpleRelationModule(nn.Module):
	def __init__(self):
		super(VerySimpleRelationModule, self).__init__()
		hidden_dim = 8
		self.layer = nn.Sequential(
			nn.Linear(2, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)

	def forward(self, combined):
		out = combined.view(combined.size(0), -1)
		out = self.layer(out)
		return out

def test3():
 
	# use dummy data
	sample = torch.arange(CLASS_IN_EP).type(torch.FloatTensor)
	sample = sample.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
	sample = sample.expand(-1, -1, 28, 28).contiguous()
	query = sample.expand(-1, QUERY_SIZE, -1, -1).contiguous()
	query = query.view(-1, 1, 28, 28)
	sample_features = (sample.expand(-1, FEATURE_DIM, -1, -1))[:, :, 0:1, 0:1] - 1
	query_features = (query.expand(-1, FEATURE_DIM, -1, -1))[:, :, 0:1, 0:1] - 1
	combined, score_target = combine_pairs(sample_features, query_features)

	# basic regression task
	X = torch.zeros(9, 2, dtype=torch.float)
	Y = torch.zeros(9, 1, dtype=torch.float)
	for i in [0, 1, 2]:
		for j in [0, 1, 2]:
			X[3*i+j, 0] = i-1
			X[3*i+j, 1] = j-1
			Y[3*i+j, 0] = 1 if i==j else 0
	#print(X)
	#print(Y)
	print(Y)
	print(score_target)

	net = SimpleRelationModule()
	criterion = nn.MSELoss()
	opt = torch.optim.Adam(net.parameters(), lr=0.01)

	for i in range(1000):
		opt.zero_grad()
		output = net(X)
		loss = criterion(output, Y)
		loss.backward()
		opt.step()

	res = output.detach().numpy()
	print(res)
	# this works



def test4():
 
	# use dummy data
	sample = torch.arange(CLASS_IN_EP).type(torch.FloatTensor)
	sample = sample.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
	sample = sample.expand(-1, -1, 28, 28).contiguous()
	query = sample.expand(-1, QUERY_SIZE, -1, -1).contiguous()
	query = query.view(-1, 1, 28, 28)
	sample_features = (sample.expand(-1, FEATURE_DIM, -1, -1))[:, :, 0:1, 0:1] - 1
	query_features = (query.expand(-1, FEATURE_DIM, -1, -1))[:, :, 0:1, 0:1] - 1
	combined, score_target = combine_pairs(sample_features, query_features)
	score_target = score_target.view(-1, 1)

	# basic regression task
	X = torch.zeros(9, 2, dtype=torch.float)
	Y = torch.zeros(9, 1, dtype=torch.float)
	for i in [0, 1, 2]:
		for j in [0, 1, 2]:
			X[3*i+j, 0] = i-1
			X[3*i+j, 1] = j-1
			Y[3*i+j, 0] = 1 if i==j else 0

	net = SimpleRelationModule()
	criterion = nn.MSELoss()
	opt = torch.optim.Adam(net.parameters(), lr=0.01)

	for i in range(1000):
		opt.zero_grad()
		output = net(X)
		loss = criterion(output, score_target)
		loss.backward()
		opt.step()

	res = output.detach().numpy()
	print(res)
	# this works


test2()
