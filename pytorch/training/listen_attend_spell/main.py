import torch
import torchaudio
import matplotlib.pyplot as plt

from model import *
from dataset import get_data_loader

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
	listener = Listen()

test2()
