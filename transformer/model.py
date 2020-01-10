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
		-----
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
		-----
		out_probs: [B, Lt, Vt]
		'''
		emb = self.embedding(target)
		emb = emb + positional_encoding(emb.shape)
		for n in range(self.n_blocks):
			emb = self.dec_blocks[n](emb)
		out_probs = self.out_layer(emb)
		return out_probs


class EncoderBlock(nn.Module):
	def __init__(self, d_model, d_ff):
		super(EncoderBlock, self).__init__()
		self.feed_forward = nn.Sequential(
			nn.Linear(d_model, d_ff),
			nn.ReLU(),
			nn.Linear(d_ff, d_model)
		)
		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)

	def forward(self, b_in):
		'''
		b_in: [B, Ls, Dm]
		-----
		b_out: [B, Ls, Dm]
		'''
		b_out = self.layernorm1(b_in)
		# TODO: specify encoder block
		return b_out


class DecoderBlock(nn.Module):
	def __init__(self, d_model, d_ff):
		super(DecoderBlock, self).__init__()
		self.feed_forward = nn.Sequential(
			nn.Linear(d_model, d_ff),
			nn.ReLU(),
			nn.Linear(d_ff, d_model)
		)
		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)
		self.layernorm3 = nn.LayerNorm(d_model)

	def forward(self, b_in):
		'''
		b_in: [B, Ls, Dm]
		-----
		b_out: [B, Ls, Dm]
		'''
		b_out = b_in
		# TODO: specify decoder block
		return b_out

class MultiHeadAttn(nn.Module):
	def __init__(self):
		super(MultiHeadAttn, self).__init__()


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