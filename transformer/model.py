import torch
import torch.nn as nn
import math
import numpy as np

'''
B: batch size
Ls: source length
Lt: target length
Vs: source vocabulary size
Vt: target vocabulary size
Dm: model dimension
'''


class Encoder(nn.Module):
	def __init__(self, n_blocks, d_model, vsize_src, d_ff):
		super(Encoder, self).__init__()
		self.n_blocks = n_blocks
		self.embedding = nn.Embedding(vsize_src, d_model)
		block_list = [EncoderBlock(d_model, d_ff) for _ in range(n_blocks)]
		self.enc_blocks = nn.ModuleList(block_list)

	def forward(self, source, src_mask):
		'''
		source: [B, Ls] (int from range(Vs))
		src_mask: [B, Ls]
		---
		enc_out: [B, Ls, Dm]
		'''
		mask = expand_mask(src_mask, Lq=None, autoreg=False)
		emb = self.embedding(source) # [B, Ls, Dm]
		emb = emb + positional_encoding(emb.shape)
		for n in range(self.n_blocks):
			emb = self.enc_blocks[n](emb, mask)
		enc_out = emb
		return enc_out


class Decoder(nn.Module):
	def __init__(self, n_blocks, d_model, vsize_tar, d_ff):
		super(Decoder, self).__init__()
		self.n_blocks = n_blocks
		self.embedding = nn.Embedding(vsize_tar, d_model)
		block_list = [DecoderBlock(d_model, d_ff) for _ in range(n_blocks)]
		self.dec_blocks = nn.ModuleList(block_list)
		self.out_layer = nn.Sequential(
			nn.Linear(d_model, vsize_tar),
			nn.Softmax(dim=2)
		)

	def forward(self, enc_out, target, src_mask, tar_mask):
		'''
		enc_out: [B, Ls, Dm]
		target: [B, Lt] (int from range(Vt))
		src_mask: [B, Ls]
		tar_mask: [B, Lt]
		---
		out_probs: [B, Lt, Vt]
		'''
		self_mask = expand_mask(tar_mask, Lq=None, autoreg=True)
		cross_mask = expand_mask(src_mask, Lq=tar_mask.size(1), autoreg=False)
		emb = self.embedding(target)
		emb = emb + positional_encoding(emb.shape)
		for n in range(self.n_blocks):
			emb = self.dec_blocks[n](emb, enc_out, self_mask, cross_mask)
		out_probs = self.out_layer(emb)
		return out_probs


class EncoderBlock(nn.Module):
	def __init__(self, d_model, d_ff):
		super(EncoderBlock, self).__init__()
		self.feedforward = nn.Sequential(
			nn.Linear(d_model, d_ff),
			nn.ReLU(),
			nn.Linear(d_ff, d_model)
		)
		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)
		self.multihead = MultiHeadAttn(d_model)

	def forward(self, b_in, mask):
		'''
		b_in: [B, Ls, Dm]
		mask: [B, Ls, Ls]
		---
		b_out: [B, Ls, Dm]
		'''
		z1 = self.multihead(b_in, b_in, b_in, mask)
		a1 = self.layernorm1(z1 + b_in)
		z2 = self.feedforward(a1)
		b_out = self.layernorm2(z2 + a1)
		return b_out


class DecoderBlock(nn.Module):
	def __init__(self, d_model, d_ff):
		super(DecoderBlock, self).__init__()
		self.feedforward = nn.Sequential(
			nn.Linear(d_model, d_ff),
			nn.ReLU(),
			nn.Linear(d_ff, d_model)
		)
		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)
		self.layernorm3 = nn.LayerNorm(d_model)
		self.self_multihead = MultiHeadAttn(d_model)
		self.cross_multihead = MultiHeadAttn(d_model)

	def forward(self, b_in, enc_out, self_mask, cross_mask):
		'''
		b_in: [B, Ls, Dm]
		self_mask: [B, Lt, Lt]
		cross_mask: [B, Lt, Ls]
		---
		b_out: [B, Ls, Dm]
		'''
		b_out = b_in
		z1 = self.self_multihead(b_in, b_in, b_in, self_mask)
		a1 = self.layernorm1(z1 + b_in)
		z2 = self.cross_multihead(a1, enc_out, enc_out, cross_mask)
		a2 = self.layernorm2(z2 + a1)
		z3 = self.feedforward(a2)
		b_out = self.layernorm3(z3 + a2)
		return b_out


class MultiHeadAttn(nn.Module):
	def __init__(self, d_model):
		super(MultiHeadAttn, self).__init__()
		n_heads = 8
		self.n_heads = n_heads
		d_k = d_model//n_heads
		d_v = d_k
		self.Wqs = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(n_heads)])
		self.Wks = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(n_heads)])
		self.Wvs = nn.ModuleList([nn.Linear(d_model, d_v, bias=False) for _ in range(n_heads)])
		self.Wo = nn.Linear(n_heads*d_v, d_model, bias=False)

	def forward(self, Q, K, V, mask):
		'''
		Q: [B, Lq, Dm]
		K: [B, Lv, Dm]
		V: [B, Lv, Dm]
		mask: [B, Lq, Lv]
		---
		out: [B, Lq, Dm]
		'''
		heads = []
		for i in range(self.n_heads):
			head_i = attention(self.Wqs[i](Q), self.Wks[i](K), self.Wvs[i](V), mask)
			heads.append(head_i) # [B, Lq, d_v] each
		heads_concat = torch.cat(heads, dim=2) # [B, Lq, d_v*h]
		out = self.Wo(heads_concat) # [B, Lq, Dm]
		return out


def attention(Q, K, V, mask):
	'''
	Q: [B, Lq, Dq]
	K: [B, Lv, Dq]
	V: [B, Lv, Dv]
	mask: [B, Lq, Lv] (1: mask, 0: no mask)
	---
	out: [B, Lq, Dv]
	'''
	scores = torch.einsum('bqd,bvd->bqv', Q, K) # [B, Lq, Lv]
	scores[mask] = float('-inf')
	weights = nn.functional.softmax(scores/math.sqrt(K.size(2)), dim=2)
	out = torch.einsum('bqv,bvd->bqd', weights, V) # [B, Lq, Dv]
	return out


def positional_encoding(shape):
	# TODO: calculate positional encoding
	pos_enc = torch.zeros(shape)
	return pos_enc


def test():

	n_blocks = 6
	d_model = 512
	vsize_src = 100
	vsize_tar = 128
	d_ff = 2048
	enc = Encoder(n_blocks, d_model, vsize_src, d_ff)
	dec = Decoder(n_blocks, d_model, vsize_tar, d_ff)

	batch_size=4
	len_src=10
	len_tar=8
	source = torch.zeros([batch_size, len_src], dtype=torch.long)
	target = torch.zeros([batch_size, len_tar], dtype=torch.long)
	src_mask = torch.zeros([batch_size, len_src], dtype=torch.bool)
	tar_mask = torch.zeros([batch_size, len_tar], dtype=torch.bool)
	for b in range(batch_size):
		src_mask[b, np.random.randint(len_src//2, len_src):] = 1
		tar_mask[b, np.random.randint(len_tar//2, len_tar):] = 1
	print(source.shape)
	enc_out = enc(source, src_mask)
	out_probs = dec(enc_out, target, src_mask, tar_mask)
	print(out_probs.shape)


def test2():
	# masking test
	batch_size=4
	len_src=10
	len_tar=8
	source = torch.zeros([batch_size, len_src], dtype=torch.long)
	target = torch.zeros([batch_size, len_tar], dtype=torch.long)
	src_mask = torch.zeros([batch_size, len_src], dtype=torch.bool)
	tar_mask = torch.zeros([batch_size, len_tar], dtype=torch.bool)
	for b in range(batch_size):
		src_mask[b, np.random.randint(len_src//2, len_src):] = 1
		tar_mask[b, np.random.randint(len_tar//2, len_tar):] = 1
	enc_self = expand_mask(src_mask, Lq=None, autoreg=False)
	dec_self = expand_mask(tar_mask, Lq=None, autoreg=True)
	dec_cross = expand_mask(src_mask, Lq=len_tar, autoreg=False)
	#print(dec_cross)


def expand_mask(mask2d, Lq=None, autoreg=False):
	'''
	mask2d: [B, Lv]
	Lq: int
	autoreg: bool
	---
	mask3d: [B, Lq, Lv]
	'''
	if Lq is None:
		# self attention
		L = mask2d.size(1)
		mask3d = torch.unsqueeze(mask2d, dim=1).expand((-1, L, -1))
		if autoreg:
			# decoder self attention
			automask = 1-torch.tril(torch.ones((L, L), dtype=torch.uint8)) # [Lq, Lv]
			automask = automask.to(torch.bool)
			mask3d = automask + mask3d
		else:
			# encoder self attention
			pass
	else:
		# cross attention
		mask3d = torch.unsqueeze(mask2d, dim=1).expand((-1, Lq, -1))
	return mask3d


test()