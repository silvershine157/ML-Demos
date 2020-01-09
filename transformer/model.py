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
	def __init__(self):
		super(Encoder, self).__init__()
		# TODO: specify encoder

	def forward(self, source):
		'''
		source: [B, Ls] (int from range(Vs))
		-----
		enc_out: [B, Ls, Dm]
		'''
		return enc_out

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		# TODO: specify decoder

	def forward(self, enc_out, target):
		'''
		enc_out: [B, Ls, Dm]
		target: [B, Lt] (int from range(Vt))
		-----
		out_probs: [B, Lt, Vt]
		'''
		return out_probs
