import numpy as np
import torchvision
import torch
import torch.nn as nn
import random
import itertools
from constants import device, PAD_token, START_token, END_token

## Show, Attend and Tell Model

class InitStateMLP(nn.Module):

	def __init__(self, a_dim, cell_dim):
		super(InitStateMLP, self).__init__()

		# dimensions
		self.a_dim = a_dim # 'D'
		self.cell_dim = cell_dim # 'n'

		# layers
		# 'f_init,c' and 'f_init,h': n and n <- D
		self.init_mlp = nn.Sequential(
			nn.Linear(self.a_dim, 200),
			nn.ReLU(),
			nn.Linear(200, 2 * self.cell_dim)
		)

	def forward(self, annotations):

		avg_anno = annotations.mean(2) # average spatially
		out = self.init_mlp(avg_anno)

		# (1, B, n)
		init_memory = out[:, :self.cell_dim].unsqueeze(dim=0)
		init_hidden = out[:, self.cell_dim:].unsqueeze(dim=0)

		return init_memory, init_hidden


class SoftAttention(nn.Module):

	def __init__(self, a_dim, a_size, cell_dim):
		super(SoftAttention, self).__init__()

		# dimensions
		self.a_dim = a_dim # 'D'
		self.a_size = a_size # 'L'
		self.cell_dim = cell_dim # 'n'

		# layers

		#'f_att': scalar <- D + n
		self.scoring_mlp = nn.Sequential(
			nn.Linear(self.a_dim + self.cell_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 1)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, annotations, last_memory):

		# annotations: (B, D, L)
		# last_memory: (1, B, n)

		# (B, n, L)
		expanded = last_memory.squeeze(dim=0).unsqueeze(dim=2).expand((-1, -1, self.a_size))

		# (B, D+n, L)
		anno_and_mem = torch.cat((annotations, expanded), dim=1)
		
		# (B, L, D+n)
		anno_and_mem = anno_and_mem.transpose(1, 2)

		# (B, L)
		scores = self.scoring_mlp(anno_and_mem).squeeze(dim=2)
		attn_weights = self.softmax(scores)

		# (B, D)
		context = torch.einsum('bdl,bl->bd', annotations, attn_weights)

		# also return weights to enable doubly stochastic attn
		return context, attn_weights


class ContextDecoder(nn.Module):

	def __init__(self, voc_size, embedding_dim, cell_dim, a_dim):
		super(ContextDecoder, self).__init__()

		# dimensions
		self.voc_size = voc_size # 'K'
		self.embedding_dim = embedding_dim # 'm'
		self.cell_dim = cell_dim # 'n'
		self.a_dim = a_dim # 'D'

		# layers

		# 'E': m <- K
		self.embedding_layer = nn.Embedding(self.voc_size, self.embedding_dim)

		# input: D + m, cell dim: n
		self.lstm_cell = nn.LSTM(self.a_dim + self.embedding_dim, self.cell_dim)

		# 'L_h': m <- n
		self.out_state_layer = nn.Linear(self.cell_dim, self.embedding_dim)

		# 'L_z': m <- D
		self.out_context_layer = nn.Linear(self.a_dim, self.embedding_dim)

		# 'L_o': K <- m
		self.out_final_layer = nn.Linear(self.embedding_dim, self.voc_size)
		self.out_softmax = nn.Softmax(dim=1)


	def forward(self, context, input_word, last_hidden, last_memory):

		# (1, B, D)
		context = context.unsqueeze(dim=0)

		# (1, B, m)
		embedded = self.embedding_layer(input_word)

		# (1, B, m + D)
		cell_input = torch.cat((context, embedded), dim=2)

		last_hidden = last_hidden.contiguous()
		last_memory = last_memory.contiguous()

		_, (hidden, memory) = self.lstm_cell(cell_input, (last_hidden, last_memory))
		# hidden: (1, B, n)
		# memory: (1, B, n)
		# output is same as hidden for single-step computation

		# (1, B, m)
		out_src = embedded + self.out_state_layer(hidden) + self.out_context_layer(context)
		
		# (B, K)
		out_scores = self.out_final_layer(out_src).squeeze(0)
		probs = self.out_softmax(out_scores)

		return probs, hidden, memory


# negative log likelihood (cross entropy) loss for decoder output
def maskNLLLoss(probs, target, mask):

	# probs: (B, K)
	# target: (1, B)
	# mask: (1, B)
	nTotal = mask.sum()
	crossEntropy = -torch.log(torch.gather(probs, 1, target.view(-1, 1)).squeeze(1))

	loss = crossEntropy.masked_select(mask).mean()
	loss = loss.to(device)

	return loss, nTotal.item()


class SoftSATModel(nn.Module):

	def __init__(self, cnn_activations_shape, voc_size, embedding_dim, cell_dim):
		super(SoftSATModel, self).__init__()

		# dimensions
		_, a_dim, a_W, a_H = list(cnn_activations_shape)
		self.a_dim = a_dim
		self.a_size = a_W * a_H
		self.voc_size = voc_size
		self.embedding_dim = embedding_dim
		self.cell_dim = cell_dim

		# submodules
		self.init_mlp = InitStateMLP(self.a_dim, self.cell_dim)
		self.soft_attn = SoftAttention(self.a_dim, self.a_size, self.cell_dim)
		self.decoder = ContextDecoder(self.voc_size, self.embedding_dim, self.cell_dim, self.a_dim)


	def forward(self, batch):
		annotation_batch, caption_batch, mask_batch, max_target_len = batch
		# caption_batch: (max_target_len, B)
		# annotation_batch: (B, D, L)
		batch_size = annotation_batch.size(0)

		annotation_batch = annotation_batch.to(device)
		caption_batch = caption_batch.to(device)
		mask_batch = mask_batch.to(device)

		# initialize
		init_memory, init_hidden = self.init_mlp(annotation_batch)	
		decoder_memory = init_memory
		decoder_hidden = init_hidden
		decoder_input = torch.LongTensor([[START_token for _ in range(batch_size)]])
		decoder_input = decoder_input.to(device)
		loss = 0.0
		rectified_losses = []
		n_total = 0

		# use teacher forcing
		for t in range(max_target_len):
			# get context vector
			context, attn_weights = self.soft_attn(annotation_batch, decoder_memory)

			# decoder forward
			probs, decoder_hidden, decoder_memory = self.decoder(
				context, decoder_input, decoder_hidden, decoder_memory
			)

			# teacher forcing
			# (1, B)
			decoder_input = caption_batch[t].view(1, -1)

			mask_loss, nTotal = maskNLLLoss(probs, caption_batch[t], mask_batch[t])
			loss += mask_loss
			n_total += nTotal
			rectified_losses.append(mask_loss.item() * nTotal)


		return loss, sum(rectified_losses)/n_total


	def greedy_decoder(self, annotation_batch, max_len):

		# initialize
		annotation_batch = annotation_batch.to(device)
		init_memory, init_hidden = self.init_mlp(annotation_batch)	
		decoder_memory = init_memory
		decoder_hidden = init_hidden
		batch_size = annotation_batch.size(0)
		decoder_input = torch.LongTensor([[START_token for _ in range(batch_size)]])
		decoder_input = decoder_input.to(device)
		all_tokens = torch.zeros([0], device=device, dtype=torch.long)

		# greedy decoding
		for _ in range(max_len):
			context, attn_weights = self.soft_attn(annotation_batch, decoder_memory)
			probs, decoder_hidden, decoder_memory = self.decoder(
				context, decoder_input, decoder_hidden, decoder_memory
			)
			_, decoder_input = torch.max(probs, dim=1)
			decoder_input = torch.unsqueeze(decoder_input, 0)
			all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

		# (max_len, B)
		all_tokens = all_tokens.transpose(0, 1)

		return all_tokens



