import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os

class SpeechDataset(Dataset):
	def __init__(self, root_dir):
		self.metadata = self.scan_metadata(root_dir)

	def __getitem__(self, idx):
		audio_filename, text = self.metadata[idx]
		audio = None
		text = None
		item = {"audio":audio, "text":text}
		return item
	
	# read text data and audio filenames 
	def scan_metadata(self, root_dir):
		metadata = []
		for speaker in os.listdir(root_dir):
			speaker_dir = os.path.join(root_dir, speaker)
			for chapter in os.listdir(speaker_dir):
				chapter_dir = os.path.join(speaker_dir, chapter)
				chapter_audio = {}
				for fname in os.listdir(chapter_dir):
					fpath = os.path.join(chapter_dir, fname)
					if fname[-4:] == ".txt":
						chapter_text = self.parse_text(fpath)
					elif fname[-5:] == ".flac":
						chapter_audio[fname[:-5]] = fpath
				for k in chapter_text:
					data_pair = (chapter_audio[k], chapter_text[k])
					metadata.append(data_pair)
		return metadata

	def parse_text(self, textfile):
		chapter_text = {}
		with open(textfile, 'r') as f:
			for line in f:
				L = line.strip().split()
				text = ' '.join(L[1:])
				chapter_text[L[0]] = text
		return chapter_text

def test():
	root_dir = "data/LibriSpeech/dev-clean/"
	dataset = SpeechDataset(root_dir)
	for pair in dataset.metadata:
		print(pair[0])
		print(pair[1])

test()
