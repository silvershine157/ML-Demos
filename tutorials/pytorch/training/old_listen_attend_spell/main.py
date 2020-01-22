import torch
import torchaudio
import matplotlib.pyplot as plt

from model import *
from dataset import get_data_loader, voc_size

# data -> log-mel spectogram -> augmentation -> CNN -> encoder -> attn decoder

visual_path = "data/visual/"

'''
plt.figure()
plt.imshow(specgram.log2()[0, : ,:].detach().numpy(), cmap='gray')
plt.savefig(visual_path + "specgram.png")
'''

def test1():
	# visualize spectrograms in a batch
	root_dir = "data/LibriSpeech/dev-clean"
	loader = get_data_loader(root_dir, batch_size=4, shuffle=True)
	for i, batch in enumerate(loader):
		spec = batch["specgram"]
		break
	fig = plt.figure()
	for i in range(4):
		ax = fig.add_subplot(4, 1, i+1)
		ax.imshow(spec.log2()[i, :, :].detach().numpy(), cmap='gray')
	plt.savefig(visual_path + "specgram.png")

def test2():
	# LAS forward pass
	root_dir = "data/LibriSpeech/dev-clean"
	loader = get_data_loader(root_dir, batch_size=4, shuffle=True)
	for i, batch in enumerate(loader):
		break
	spec = batch["specgram"]
	print(spec.size())
	h_dim=512
	listen = Listen(n_mels=40, h_dim=h_dim)
	spell = AttendAndSpell(h_dim=h_dim, voc_size=voc_size())
	h = listen(spec)
	print(h.size())
	spell(h)


def test3():
	# reducing time resolution
	x = torch.arange(16*3)
	x = x.view(3, 16).permute(1, 0)
	x = x.unsqueeze(dim=2).expand(-1, -1, 2)
	print(x[0, :, :])
	print(x[-1, :, :])
	print(x.size())
	x = x.view(-1, 2, 3, 2)
	x = torch.cat((x[:, 0, :, :], x[:, 1, :, :]), dim=2)
	print(x.size())
	print(x[0, :, :])
	print(x[-1, :, :])
	pass

test2()
