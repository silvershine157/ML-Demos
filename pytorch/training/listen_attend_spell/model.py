import torch
import torch.nn as nn

class Listen(nn.Module):
	def __init__(self, n_mels, h_dim):
		super(Listen, self).__init__()
		self.h_dim=h_dim
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


class Spell(nn.Module):
	def __init__(self, h_dim, voc_size):
		super(Spell, self).__init__()
		self.h_dim = h_dim
		self.embed_dim = 100
		self.input_dim = self.h_dim + self.embed_dim
		self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=h_dim, num_layers=2, bidirectional=False)
		self.voc_size = voc_size
		self.embedding = nn.Embedding(self.voc_size, self.embed_dim)
		self.out_layer = nn.Sequential(
			nn.Linear(h_dim, voc_size),
			nn.Softmax(dim=2) # [1, B, voc_size]
		)

	def forward(self, h):
		"""
		Args:
			h: high level representation [B, h_dim, max_U]
		Returns:
			out_probs: output probabilities [B, max_S, alphabet_size]
		"""
		B = h.size(0)
		max_characters = 3
		# any better way?
		hidden_state = torch.zeros((2, B, self.h_dim))
		cell_state = torch.zeros((2, B, self.h_dim))
		SOS_token = 1 # TODO: make Voc class
		last_token = torch.LongTensor([SOS_token for _ in range(B)])
		for i in range(max_characters):
			context = torch.zeros(B, self.h_dim)
			# TODO: context = attn(last_state, h)
			last_embedding = self.embedding(last_token)
			cell_input = torch.cat((last_embedding, context), dim=1).unsqueeze(dim=0)
			report(cell_input)
			hidden_state, cell_state = self.lstm(cell_input, (hidden_state, cell_state))
			report(hidden_state)
			out_distr = self.out_layer(hidden_state)
			report(out_distr)
			best_token = out_distr.argmax(dim=3)
			break
		pass

def report(tensor):
	print(tensor.size(), tensor.dtype)

class LAS(nn.Module):
	def __init__(self):
		super(LAS, self).__init__()
		self.listener = Listen()
		self.speller = Spell()

	def forward(self, spec, target=None, beta=1):
		"""
		Args:
			spec: specgram image [B, N_MELS, max_T]
			target: indices of target sequence [B, max_S], None for test mode
			beta: number of partial hypothesis to keep in beam search
		Returns:
			out_probs: output probabilities [B, max_S, voc_size]
			out_labels: output labels [B, max_S, voc_size]
		"""
		h = self.listener(spec)
		if target:
			# train mode
			out_probs = self.decoder(h, None)
			loss = cross_ent_loss(target, out_probs)			
			pass
		else:
			# test mode
			beams = ["SOS" for _ in range(beta)]
			max_seq_len = 20
			for _ in range(max_seq_len):
				next_beams = []
				for beam in beams:
					next_probs = self.decoder(h, beam)
					next_beams.extend(expand_beam(beam, next_probs))
				beams = prune_beams(next_beams, beta)

		return out_probs

