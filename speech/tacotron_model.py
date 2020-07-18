import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import textproc
from const import *

class Tacotron2(nn.Module):
	def __init__(self):
		super(Tacotron2, self).__init__()
		self.d_enc = 512
		self.d_context = 128
		self.d_spec = 552

	def forward(self, token_pad, token_lengths, S_true, teacher_forcing):
		'''
		token_pad: [Lt_max, B]
		token_lengths: [B]
		S_true: [St_max, B, d_spec] or None
		---
		S_before: [max_dec_len, B, d_spec]
		S_after: [max_dec_len, B, d_spec]
		stop_logits: [max_dec_len, B]
		'''
		return S_before, S_after

class Encoder(nn.Module):
	def __init__(self, d_enc):
		super(Encoder, self).__init__()
		self.d_enc = d_enc

	def forward(self, token_pad, token_lengths):
		'''
		token_pad: [Lt_max, B]
		token_lengths: [B]
		---
		enc_out: [B, d_enc]
		'''
		return enc_out

class AttnDecoder(nn.Module):
	def __init__(self, d_enc, d_context, d_spec):
		super(AttnDecoder, self).__init__()

	def forward(self, enc_out, S_true, teacher_forcing):
		'''
		enc_out: [B, d_enc]
		S_true: [St_max, B, d_spec] or None
		---
		S_before: [max_dec_len, B, d_spec]
		S_after: [max_dec_len, B, d_spec]
		stop_logits: [max_dec_len, B]
		'''
		return S_before, S_after, stop_logits
