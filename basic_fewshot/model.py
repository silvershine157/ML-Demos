import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingNetwork(nn.Module):
	def __init__(self):
		super(MatchingNetwork, self).__init__()
		d_embed = 64
		self.support_enc = EncoderCNN(d_embed)
		self.query_enc = EncoderCNN(d_embed)

	def loss(self, support, query, label):
		'''
		support: [1, C, K, 1, 105, 105]
		query: [1, Q, 1, 105, 105]
		label: [1, Q]
		'''
		C = support.size(1)
		K = support.size(2)
		Q = query.size(1)
		flat_support = support.view((-1, 1, 105, 105))
		h_s = self.support_enc(flat_support).view((C, K, -1)) # [C, K, d_embed]
		h_q = self.query_enc(query.squeeze(0)) # [Q, d_embed]
		h_s_expand = h_s.unsqueeze(0).expand((Q, -1, -1, -1))
		h_q_expand = h_q.unsqueeze(1).unsqueeze(2).expand((-1, C, K, -1))
		cos_sim = F.cosine_similarity(h_s_expand, h_q_expand, dim=3) # [Q, C, K]
		out = F.normalize(torch.sum(torch.exp(cos_sim), dim=2), p=1, dim=1) # [Q, C]
		loss = F.nll_loss(torch.log(out), label.squeeze(dim=0))
		return loss

	def infer(self, support, query):
		'''
		support: [1, C, K, 1, 105, 105]
		query: [1, Q, 1, 105, 105]
		---
		pred: [1, Q]
		'''
		C = support.size(1)
		K = support.size(2)
		Q = query.size(1)
		flat_support = support.view((-1, 1, 105, 105))
		h_s = self.support_enc(flat_support).view((C, K, -1)) # [C, K, d_embed]
		h_q = self.query_enc(query.squeeze(0)) # [Q, d_embed]
		h_s_expand = h_s.unsqueeze(0).expand((Q, -1, -1, -1))
		h_q_expand = h_q.unsqueeze(1).unsqueeze(2).expand((-1, C, K, -1))
		cos_sim = F.cosine_similarity(h_s_expand, h_q_expand, dim=3) # [Q, C, K]
		score_matrix = torch.sum(torch.exp(cos_sim), dim=2) # [Q, C]
		pred = torch.argmax(score_matrix, dim=1).unsqueeze(0) # [1, Q]
		return pred

class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		d_embed = 64
		self.encoder = EncoderCNN(d_embed)
		self.distance_weights = nn.Linear(d_embed, 1)

	def prob_same_class(self, h0, h1):
		'''
		h0,1: [shape, d_embed]
		---
		p: [shape]
		'''
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
		h0 = self.encoder(img0)
		h1 = self.encoder(img1)
		p = self.prob_same_class(h0, h1)
		loss = F.binary_cross_entropy(p, label)
		return loss

	# C-way K-shot inference
	def infer(self, support, query):
		'''
		support: [1, C, K, 1, 105, 105]
		query: [1, Q, 1, 105, 105]
		---
		pred: [1, Q]
		'''
		C = support.size(1)
		K = support.size(2)
		Q = query.size(1)
		flat_support = support.view((-1, 1, 105, 105))
		h_s = self.encoder(flat_support).view((C, K, -1)) # [C, K, d_embed]
		h_q = self.encoder(query.squeeze(0)) # [Q, d_embed]
		h_s_expand = h_s.unsqueeze(0).expand((Q, -1, -1, -1))
		h_q_expand = h_q.unsqueeze(1).unsqueeze(2).expand((-1, C, K, -1))
		p_tensor = self.prob_same_class(h_s_expand, h_q_expand).squeeze(3) # [Q, C, K]
		p_matrix = torch.mean(p_tensor, dim=2) # [Q, C]
		pred = torch.argmax(p_matrix, dim=1).unsqueeze(0) # [1, Q]
		return pred


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