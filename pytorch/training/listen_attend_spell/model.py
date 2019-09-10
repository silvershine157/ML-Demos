import torch
import torch.nn as nn

class Listen(nn.Module):
	def __init__(self, n_mels):
		super(Listen, self).__init__()
		self.h_dim=512
		self.blstm = nn.LSTM(input_size=n_mels, hidden_size=self.h_dim//2, num_layers=1, bidirectional=True)
		self.pblstms = []
		for i in range(3):
			self.pblstms.append(nn.LSTM(input_size=self.h_dim*2, hidden_size=self.h_dim//2, num_layers=1, bidirectional=True))

	def forward(self, spec):
		"""
		Args:
			spec: specgram image [B, N_MELS, max_T]
		Returns:
			h: high level representation [B, h_dim, max_U]
		"""
		B = spec.size(0)

		# transpose to [max_T, B, N_MELS]
		listen_input = spec.permute(2, 0, 1)
		out, _ = self.blstm(listen_input) # [max_T, B, h_dim]
		for j in range(len(self.pblstms)):
			out = out.view(-1, 2, B, self.h_dim) # split even and odd
			reduced = torch.cat((out[:,0,:,:], out[:,1,:,:]), dim=2)
			# reduced: [(half), B, 2*h_dim]
			out, _ = self.pblstms[j](reduced)
		
		# retranspose to [B, h_dim, max_U]
		h = out.permute(1, 2, 0)
		return h


class AttendAndSpell(nn.Module):
	def __init__(self):
		super(AttendAndSpell, self).__init__()

	def forward(self, h):
		"""
		Args:
			h: high level representation [B, h_dim, max_U]
		Returns:
			out_probs: output probabilities [B, max_S, alphabet_size]
		"""
		pass

class LAS(nn.Module):
	def __init__(self):
		super(LAS, self).__init__()
		self.listener = Listen()
		self.speller = AttendAndSpell()

	def forward(self, spec):
		"""
		Args:
			spec: specgram image [B, N_MELS, max_T]
		Returns:
			out_probs: output probabilities [B, max_S, alphabet_size]
		"""
		h = self.listener(spec)
		out_probs = self.speller(h)
		return out_probs

