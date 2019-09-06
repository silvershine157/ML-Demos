import torch.nn as nn

class Listen(nn.Module):
	def __init__(self):
		super(Listen, self).__init__()

	def forward(self, spec):
		"""
		Args:
			spec: specgram image [B, N_MELS, max_T]
		Returns:
			h: high level representation [B, h_dim, max_U]
		"""

		pass

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

