import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import textproc

class MiniTTS(nn.Module):
	# simplest possible model for linear spectrogram TTS
	def __init__(self):
		super(MiniTTS, self).__init__()
		d_hidden = 256
		d_spec = 552
		self.embedding = nn.Embedding(len(textproc.symbols), d_hidden)
		self.enc_rnn = nn.LSTM(d_hidden, d_hidden)
		self.dec_rnn = nn.LSTM(d_hidden, d_hidden)
		self.spec_pre = nn.Linear(d_spec, d_hidden)
		self.spec_post = nn.Linear(d_hidden, d_spec)
		self.stop_layer = nn.Linear(d_hidden, 1)

	def forward(self, token_pad, token_lengths):
		'''
		token_pad: [Lt_max, B]
		token_lenghts: [B]
		'''
		B = token_pad.shape[1]
		token_emb = self.embedding(token_pad) # [Lt_max, B, d_hidden]
		token_pack = pack_padded_sequence(token_emb, token_lengths, enforce_sorted=False)
		h_0 = torch.zeros(token_emb.shape[1:])
		_, (h_n, c_n) = self.enc_rnn(token_pack)
		enc_out = h_n[-1, :, :] # [B, d_hidden]
		

		return None