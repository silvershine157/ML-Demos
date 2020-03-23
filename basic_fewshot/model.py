import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		d_embed = 64
		self.encoder = EncoderCNN(d_embed)
		self.distance_weights = nn.Linear(d_embed, 1)

	def prob_same_class(self, img0, img1):
		'''
		img0,1: [B, 1, 105, 105]
		---
		p: [B]
		'''
		h0 = self.encoder(img0)
		h1 = self.encoder(img1)
		p = torch.sigmoid(self.distance_weights(torch.abs(h0 - h1)))
		p = p.squeeze(dim=1)
		return p

	def loss(self, pair, label):
		'''
		pair: [B, 2, 1, 105, 105]
		label: [B]
		'''
		img0 = pair[:, 0, :, :, :]
		img1 = pair[:, 1, :, :, :]
		p = self.prob_same_class(img0, img1)
		loss = F.binary_cross_entropy(p, label)
		return loss

class EncoderCNN(nn.Module):
	def __init__(self, d_embed):
		super(EncoderCNN, self).__init__()
		self.layers = nn.Sequential( # (105, 105)
			nn.Conv2d(1, 32, 3),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2), # (51, 51)
			nn.Conv2d(32, 32, 3),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2), # (24, 24)
			nn.Conv2d(32, 32, 3),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2), # (11, 11)
			nn.Conv2d(32, 32, 3),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2) # (4, 4)
		)
		self.fc = nn.Linear(32*4*4, d_embed)

	def forward(self, img):
		'''
		img: [B, 1, 105, 105]
		---
		h: [B, d_embed]
		'''
		B = img.size(0)
		features_cnn = self.layers(img)
		features_flat = features_cnn.view(B, -1)
		h = self.fc(features_flat)
		return h