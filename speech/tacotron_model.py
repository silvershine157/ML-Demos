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
		self.voc_size = len(textproc.symbols)

		self.encoder = Encoder(self.d_enc, self.voc_size)
		self.decoder = AttnDecoder(self.d_enc, self.d_context, self.d_spec)

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
		enc_out = self.encoder(token_pad, token_lengths)
		S_before, S_after, stop_logits = self.decoder(enc_out, S_true, teacher_forcing)

		return S_before, S_after, stop_logits

class Encoder(nn.Module):
	def __init__(self, d_enc, voc_size):
		super(Encoder, self).__init__()
		self.d_enc = d_enc
		self.char_emb = nn.Embedding(voc_size, d_enc)
		kernel_size = 5
		padding = kernel_size//2
		self.conv = nn.Sequential(
			nn.Conv1d(d_enc, d_enc, kernel_size, padding=padding),
			nn.BatchNorm1d(d_enc),
			nn.ReLU(),
			nn.Conv1d(d_enc, d_enc, kernel_size, padding=padding),
			nn.BatchNorm1d(d_enc),
			nn.ReLU(),
			nn.Conv1d(d_enc, d_enc, kernel_size, padding=padding),
			nn.BatchNorm1d(d_enc)
		)
		self.blstm = nn.LSTM(d_enc, d_enc//2, bidirectional=True)

	def forward(self, token_pad, token_lengths):
		'''
		token_pad: [Lt_max, B] (0 ~ voc_size-1)
		token_lengths: [B]
		---
		enc_out: [Lt_max, B, d_enc]
		'''
		emb = self.char_emb(token_pad) # [Lt_max, B, d_enc]
		emb = emb.transpose(0, 1).transpose(1, 2) # [B, d_enc, Lt_max]
		conv_out = self.conv(emb) # [B, d_enc, Lt_max]
		conv_out = conv_out.transpose(1, 2).transpose(0, 1) # [Lt_max, B, d_enc]
		blstm_input = pack_padded_sequence(conv_out, token_lengths, enforce_sorted=False)
		blstm_output, _ = self.blstm(blstm_input)
		enc_out, _ = pad_packed_sequence(blstm_output)

		return enc_out

class AttnDecoder(nn.Module):
	def __init__(self, d_enc, d_context, d_spec):
		super(AttnDecoder, self).__init__()

	def forward(self, enc_out, S_true, teacher_forcing):
		'''
		enc_out: [Lt_max, B, d_enc]
		S_true: [St_max, B, d_spec] or None
		---
		S_before: [max_dec_len, B, d_spec]
		S_after: [max_dec_len, B, d_spec]
		stop_logits: [max_dec_len, B]
		'''
		return S_before, S_after, stop_logits
