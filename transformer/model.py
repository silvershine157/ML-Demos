import torch
import torch.nn as nn

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

	def forward(self, source):
		'''
		source: [B, Ls] (int from range(Vs))
		---
		enc_out: [B, Ls, Dm]
		'''
		emb = self.embedding(source) # [B, Ls, Dm]
		emb = emb + positional_encoding(emb.shape)
		for n in range(self.n_blocks):
			emb = self.enc_blocks[n](emb)
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

	def forward(self, enc_out, target):
		'''
		enc_out: [B, Ls, Dm]
		target: [B, Lt] (int from range(Vt))
		---
		out_probs: [B, Lt, Vt]
		'''
		emb = self.embedding(target)
		emb = emb + positional_encoding(emb.shape)
		for n in range(self.n_blocks):
			emb = self.dec_blocks[n](emb, enc_out)
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

	def forward(self, b_in):
		'''
		b_in: [B, Ls, Dm]
		---
		b_out: [B, Ls, Dm]
		'''
		z1 = self.multihead(b_in, b_in, b_in)
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
		self.masked_multihead = MultiHeadAttn(d_model) # TODO: provide option
		self.cross_multihead = MultiHeadAttn(d_model)

	def forward(self, b_in, enc_out):
		'''
		b_in: [B, Ls, Dm]
		---
		b_out: [B, Ls, Dm]
		'''
		b_out = b_in
		z1 = self.masked_multihead(b_in, b_in, b_in)
		a1 = self.layernorm1(z1 + b_in)
		z2 = self.cross_multihead(a1, enc_out, enc_out)
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

	def forward(self, Q, K, V):
		'''
		Q: [B, Lq, Dm]
		K: [B, Lv, Dm]
		V: [B, Lv, Dm]
		---
		out: [B, Lq, Dm]
		'''
		heads = []
		for i in range(self.n_heads):
			head_i = attention(self.Wqs[i](Q), self.Wks[i](K), self.Wvs[i](V)) 
			heads.append(head_i) # [B, Lq, d_v] each
		heads_concat = torch.cat(heads, dim=2) # [B, Lq, d_v*h]
		out = self.Wo(heads_concat) # [B, Lq, Dm]
		return out


def attention(Q, K, V):
	'''
	Q: [B, Lq, Dq]
	K: [B, Lv, Dq]
	V: [B, Lv, Dv]
	---
	out: [B, Lq, Dv]
	'''
	out = Q
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

	batch_size=3
	len_src=10
	len_tar=16
	source = torch.zeros([batch_size, len_src], dtype=torch.long)
	target = torch.zeros([batch_size, len_tar], dtype=torch.long)
	
	print(source.shape)
	enc_out = enc(source)
	out_probs = dec(enc_out, target)
	print(out_probs.shape)

test()