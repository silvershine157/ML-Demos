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
		dropout_conv = 0.5
		dropout_lstm = 0.1
		dropout_prenet = 0.5
		self.encoder = Encoder(self.d_enc, self.voc_size, dropout_conv)
		self.decoder = AttnDecoder(self.d_enc, self.d_context, self.d_spec, dropout_conv, dropout_lstm, dropout_prenet)

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
	def __init__(self, d_enc, voc_size, dropout_conv):
		super(Encoder, self).__init__()
		self.d_enc = d_enc
		self.char_emb = nn.Embedding(voc_size, d_enc)
		kernel_size = 5
		padding = kernel_size//2
		self.conv = nn.Sequential(
			nn.Conv1d(d_enc, d_enc, kernel_size, padding=padding),
			nn.BatchNorm1d(d_enc),
			nn.ReLU(),
			nn.Dropout(dropout_conv),
			nn.Conv1d(d_enc, d_enc, kernel_size, padding=padding),
			nn.BatchNorm1d(d_enc),
			nn.ReLU(),
			nn.Dropout(dropout_conv),
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
	def __init__(self, d_enc, d_context, d_spec, dropout_conv, dropout_lstm, dropout_prenet):
		super(AttnDecoder, self).__init__()
		self.d_spec = d_spec
		self.d_context = d_context
		d_pre = 256
		self.pre_net = nn.Sequential(
			nn.Linear(d_spec, d_pre),
			nn.ReLU(),
			nn.Dropout(dropout_prenet),
			nn.Linear(d_pre, d_pre),
			nn.ReLU(),
			nn.Dropout(dropout_prenet)
		)
		self.max_dec_len = 1000
		self.d_lstm = 1024
		self.lstm = nn.LSTM(d_pre+d_context, self.d_lstm, num_layers=2, dropout=dropout_lstm)
		d_post = 512
		self.to_spec_frame = nn.Linear(self.d_lstm+d_context, d_spec)
		self.to_stop_logit = nn.Linear(self.d_lstm+d_context, 1)
		self.h_init = nn.Parameter(torch.zeros((2, 1, self.d_lstm), requires_grad=True))
		self.c_init = nn.Parameter(torch.zeros((2, 1, self.d_lstm), requires_grad=True))
		self.post_net = nn.Sequential(
			nn.Conv1d(d_spec, d_post, kernel_size=5, padding=2),
			nn.BatchNorm1d(d_post),
			nn.Tanh(),
			nn.Dropout(dropout_conv),
			nn.Conv1d(d_post, d_post, kernel_size=5, padding=2),
			nn.BatchNorm1d(d_post),
			nn.Tanh(),
			nn.Dropout(dropout_conv),
			nn.Conv1d(d_post, d_post, kernel_size=5, padding=2),
			nn.BatchNorm1d(d_post),
			nn.Tanh(),
			nn.Dropout(dropout_conv),
			nn.Conv1d(d_post, d_post, kernel_size=5, padding=2),
			nn.BatchNorm1d(d_post),
			nn.Tanh(),
			nn.Dropout(dropout_conv),
			nn.Conv1d(d_post, d_spec, kernel_size=5, padding=2)
		)

	def forward(self, enc_out, S_true, teacher_forcing):
		'''
		enc_out: [Lt_max, B, d_enc]
		S_true: [St_max, B, d_spec] or None
		---
		S_before: [max_dec_len, B, d_spec]
		S_after: [max_dec_len, B, d_spec]
		stop_logits: [max_dec_len, B]
		'''

		B = enc_out.shape[1]
		St_max = S_true.shape[0]
		zero_frame = torch.zeros((B, self.d_spec), device=device)
		spec_frame_list = []
		stop_logit_list = []
		lstm_state = (self.h_init.repeat(1, B, 1), self.c_init.repeat(1, B, 1))
		for t in range(self.max_dec_len):
			if t == 0:
				in_frame = zero_frame
			else:
				if teacher_forcing:
					if t < St_max:
						in_frame = S_true[t-1, :, :]
					else:
						in_frame = zero_frame
				else:
					in_frame = out_frame
			# in_frame: [B, d_spec]
			pre_out = self.pre_net(in_frame) # [B, d_pre]
			# TODO: proper attention
			context = torch.mean(enc_out, dim=0)[:, :self.d_context] 
			lstm_input = torch.cat([pre_out, context], dim=1).unsqueeze(dim=0) # [1, B, d_pre+d_context]
			lstm_output, lstm_state = self.lstm(lstm_input, lstm_state) # lstm_output: [1, B, d_lstm]
			out_context = torch.cat([lstm_output.squeeze(dim=0), context], dim=1) # [1, B, d_lstm+d_context]
			out_frame = self.to_spec_frame(out_context) # [B, d_spec]
			stop_logit = self.to_stop_logit(out_context).squeeze(dim=1) # [B, 1]
			spec_frame_list.append(out_frame)
			stop_logit_list.append(stop_logit)

		S_before = torch.stack(spec_frame_list, dim=0) # [max_dec_len, B, d_spec]
		stop_logits = torch.stack(stop_logit_list, dim=0) # [max_dec_len, B]

		post_input = S_before.transpose(0, 1).transpose(1, 2) # [B, d_spec, max_dec_len]
		post_output = self.post_net(post_input) # [B, d_spec, max_dec_len]
		S_after = S_before + post_output.transpose(1, 2).transpose(0, 1) # [max_dec_len, B, d_spec]

		return S_before, S_after, stop_logits
