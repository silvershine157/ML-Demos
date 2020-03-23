import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		d_embed = 64
		self.encoder = EncoderCNN(d_embed)

	def prob_same_class(self, x1, x2):
		p = None
		return p

	def loss(self, pair, label):
		'''
		pair: [B, 2, 1, 105, 105]
		label: [B]
		'''
		img0 = pair[:, 0, :, :, :]
		img1 = pair[:, 1, :, :, :]
		h0 = self.encoder(img0)
		#h1 = self.encoder(img1)
		loss = None
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
		print(h.shape)