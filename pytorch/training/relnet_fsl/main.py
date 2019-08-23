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

## Global config parameters

TRAIN_EPISODES = 3000
LEARNING_RATE = 0.001
VALIDATE_EVERY = 300
VAL_EPISODES = 50
TEST_EPISODES = 200
PRINT_EVERY = 100
CLASS_IN_EP = 5
QUERY_SIZE = 19
FEATURE_DIM = 64
HIDDEN_SIZE = 8


## Model

'''
network architecture and training detail is identical to the author's code:
https://github.com/floodsung/LearningToCompare_FSL/blob/master/omniglot/omniglot_train_one_shot.py
'''
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
		# out: B x FEATURE_DIM x 1 x 1
		out = self.layer1(img)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)		
		return out

class RelationModule(nn.Module):
	def __init__(self):
		super(RelationModule, self).__init__()
		
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
			nn.Linear(FEATURE_DIM, HIDDEN_SIZE),
			nn.ReLU(),
			nn.Linear(HIDDEN_SIZE, 1),
			nn.Sigmoid()
		)

	def forward(self, combined):
		# combined: B x (2*FEATURE_DIM) x 1 x 1
		# out: B x 1
		out = self.layer1(combined)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.out_layer(out)
		return out

def combine_pairs(sample_features, query_features):

	flat_size = CLASS_IN_EP * QUERY_SIZE
	num_pairs = CLASS_IN_EP * flat_size

	# generate labels
	sample_classes = torch.arange(CLASS_IN_EP) # 5
	query_classes = sample_classes.unsqueeze(dim=1).expand(-1, QUERY_SIZE) # CLASS_IN_EP x QUERY_SIZE

	# expand dimensions
	sample_classes = sample_classes.unsqueeze(dim=1).expand(-1, flat_size) # CLASS_IN_EP x flat_size	
	query_classes = query_classes.contiguous().view(1, -1).expand(5, -1) # CLASS_IN_EP x flat_size
	
	# generate target
	target = (sample_classes == query_classes).type(torch.FloatTensor)
	target = target.view(num_pairs, 1)

	# expand for pairing
	# both will have size: CLASS_IN_EP X flat_size x FEATURE_DIM x 1 x 1
	sample_features = sample_features.unsqueeze(dim=1).expand(-1, flat_size, -1, -1, -1)
	query_features = query_features.unsqueeze(dim=0).expand(CLASS_IN_EP, -1, -1, -1, -1)

	# concat in depth
	combined = torch.cat([sample_features, query_features], dim=2)
	combined = combined.view(num_pairs, 2*FEATURE_DIM, 1, 1)
	# combined: num_pairs x (2*FEATURE_DIM) x 1 x 1
	
	return combined, target


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


## Data helpers

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
	metatrain_dirs = all_classes[:num_train] # 1200
	metaval_dirs = all_classes[num_train:] # 423
	metatest_dirs = metaval_dirs # seems like the author's code is doing this

	return metatrain_dirs, metaval_dirs, metatest_dirs

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


## Train/test routines

def evaluate_accuracy(embed_net, rel_net, test_episodes, metatest_loader):

	correct = 0
	with torch.no_grad():
		for _ in range(test_episodes):
			# setup episode
			ep_data = next(iter(metatest_loader))
			sample = ep_data['sample'].to(device)
			query = ep_data['query'].to(device)
			query = query.view(-1, 1, 28, 28)
			
			# forward pass
			sample_features = embed_net(sample)		
			query_features = embed_net(query)
			combined, score_target = combine_pairs(sample_features, query_features)
			score_target = score_target.to(device)
			score_pred = rel_net(combined)
			class_target = torch.argmax(score_target.view(CLASS_IN_EP, CLASS_IN_EP * QUERY_SIZE), dim=0)
			class_pred = torch.argmax(score_pred.view(CLASS_IN_EP, CLASS_IN_EP * QUERY_SIZE), dim=0)
			
			# count correct predictions
			equals = (class_target == class_pred)
			correct += torch.sum(equals).item()
	
	accuracy = correct/CLASS_IN_EP/QUERY_SIZE/test_episodes
	return accuracy


def main():
	
	print("Load data . . .")
	metatrain_dirs, metaval_dirs, metatest_dirs = get_class_dirs()
	metatrain_dataset = OmniglotOneshotDataset(metatrain_dirs)
	metatrain_loader = DataLoader(metatrain_dataset, batch_size=CLASS_IN_EP, shuffle=True)
	metaval_dataset = OmniglotOneshotDataset(metaval_dirs)
	metaval_loader = DataLoader(metaval_dataset, batch_size=CLASS_IN_EP, shuffle=True)
	metatest_dataset = OmniglotOneshotDataset(metaval_dirs)
	metatest_loader = DataLoader(metaval_dataset, batch_size=CLASS_IN_EP, shuffle=True)

	print("Build model . . .")
	embed_net = EmbeddingModule()
	rel_net = RelationModule()

	print("Setup training . . .")
	criterion = nn.MSELoss()
	embed_opt = torch.optim.Adam(embed_net.parameters(), lr=LEARNING_RATE)
	rel_opt = torch.optim.Adam(rel_net.parameters(), lr=LEARNING_RATE)
	embed_scheduler = StepLR(embed_opt, step_size=100000, gamma=0.5)
	rel_scheduler = StepLR(rel_opt, step_size=100000, gamma=0.5)
	embed_net.apply(weights_init)
	rel_net.apply(weights_init)
	embed_net.to(device)
	rel_net.to(device)

	print("Training . . .")
	running_loss = 0.0
	for episode in range(1, TRAIN_EPISODES+1):
		
		embed_scheduler.step(episode)
		rel_scheduler.step(episode)

		# setup episode
		ep_data = next(iter(metatrain_loader))
		sample = ep_data['sample'].to(device) # CLASS_IN_EP x 1 x 28 x 28
		query = ep_data['query'].to(device) # CLASS_IN_EP x QUERY_SIZE x 28 x 28
		query = query.view(-1, 1, 28, 28) # flat_size x 1 x 28 x 28

		# forward pass
		sample_features = embed_net(sample) # CLASS_IN_EP x FEATURE_DIM x 1 x 1 (avoid redundant computation)
		query_features = embed_net(query) # flat_size x FEATURE_DIM x 1 x 1
		combined, score_target = combine_pairs(sample_features, query_features)
		score_target = score_target.to(device)
		score_pred = rel_net(combined)
		loss = criterion(score_pred, score_target)

		# backward pass & update
		rel_net.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(embed_net.parameters(), 0.5)
		nn.utils.clip_grad_norm_(rel_net.parameters(), 0.5)
		embed_opt.step()
		rel_opt.step()

		# print progress
		running_loss += loss.item()
		if episode % PRINT_EVERY == 0:
			print('Episode %d, avg loss: %f'%(episode, running_loss/PRINT_EVERY))
			running_loss = 0.0

		# validate model
		if episode % VALIDATE_EVERY == 0:
			val_accuracy = evaluate_accuracy(embed_net, rel_net, VAL_EPISODES, metaval_loader)
			print('Validation accuracy: %f'%(val_accuracy))


	print("Testing . . .")
	test_accuracy = evaluate_accuracy(embed_net, rel_net, TEST_EPISODES, metatest_loader)
	print('Test accuracy: %f'%(test_accuracy))



main()
