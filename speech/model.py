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
		self.d_hidden = d_hidden
		self.d_spec = d_spec
		self.embedding = nn.Embedding(len(textproc.symbols), d_hidden)
		self.enc_rnn = nn.LSTM(d_hidden, d_hidden)
		self.dec_rnn_cell = nn.LSTMCell(d_hidden, d_hidden)
		self.spec_pre = nn.Linear(d_spec, d_hidden)
		self.spec_post = nn.Sequential(
			nn.Linear(d_hidden, d_hidden),
			nn.ReLU(),
			nn.Linear(d_hidden, d_spec)
		)
		self.stop_layer = nn.Sequential(
			nn.Linear(d_hidden, d_hidden),
			nn.ReLU(),
			nn.Linear(d_hidden, 1)
		)

	def encoder(self, token_pad, token_lengths):
		'''
		token_pad: [Lt_max, B]
		token_lengths: [B]
		---
		enc_out: [B, d_hidden]
		'''
		B = token_pad.shape[1]
		token_emb = self.embedding(token_pad) # [Lt_max, B, d_hidden]
		token_pack = pack_padded_sequence(token_emb, token_lengths, enforce_sorted=False)
		_, (h_n, c_n) = self.enc_rnn(token_pack)
		enc_out = h_n[-1, :, :] # [B, d_hidden]
		return enc_out

	def decoder_step(self, state, S_frame_in):
		'''
		state: (h: [B, d_hidden], c: [B, d_hidden])
		S_frame_in: [B, d_spec]
		---
		next_state: (h: [B, d_hidden], c: [B, d_hidden])
		S_frame_out: [B, d_spec]
		stop_logit: [B]
		'''
		h, c = state
		dec_in = self.spec_pre(S_frame_in)
		h_next, c_next = self.dec_rnn_cell(dec_in, (h, c))
		next_state = (h_next, c_next)
		S_frame_out = self.spec_post(h_next)
		stop_logit = self.stop_layer(h_next).squeeze(1)
		return next_state, S_frame_out, stop_logit

	def decoder(self, enc_out, S_true):
		'''
		enc_out: [B, d_hidden]
		S_true: [St_max, B, d_spec] or None
		---
		S_pred: [max_dec_len, B, d_spec]
		stop_logits: [max_dec_len]
		'''
		B = enc_out.shape[0]
		state = (enc_out, enc_out)
		max_dec_len = 1000
		S_frame_out = torch.zeros(B, self.d_spec)
		S_pred_list = []
		stop_logits_list = []
		teacher_forcing = True
		len_to_pad = max_dec_len - S_true.shape[0]
		assert len_to_pad > 0
		S_true_ext = F.pad(S_true, [0, 0, 0, 0, 0, len_to_pad])
		for t in range(max_dec_len):
			if t > 0:
				if teacher_forcing:
					S_frame_in = S_true_ext[t, :, :]
				else:
					S_frame_in = S_frame_out
			else:
				S_frame_in = torch.zeros(B, self.d_spec)
			state, S_frame_out, stop_logit = self.decoder_step(state, S_frame_in)
			S_pred_list.append(S_frame_out)
			stop_logits_list.append(stop_logit)
		S_pred = torch.stack(S_pred_list)
		stop_logits = torch.stack(stop_logits_list)

		return S_pred, stop_logits