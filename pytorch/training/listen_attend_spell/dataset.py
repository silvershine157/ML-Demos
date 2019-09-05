import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os

class SpeechDataset(Dataset):
	def __init__(self, root_dir):
		self.metadata = self.scan_metadata(root_dir)

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		audio_filename, text = self.metadata[idx]
		audio_wave, fs = torchaudio.load(audio_filename)
		specgram = torchaudio.transforms.MelSpectrogram()(audio_wave)
		sample = {"specgram":specgram, "text":text}
		return sample
	
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
				chapter_text[L[0]] = str_to_indices(text)
		return chapter_text


ALL_CHARS = "___ABCDEFGHIJKLMNOPQRSTUVWXYZ '"
PAD_token = 0
SOS_token = 1
EOS_token = 2

def str_to_indices(s):
	indices = [ALL_CHARS.find(c) for c in s] + [EOS_token]
	indices = torch.ShortTensor(indices)
	return indices

def indices_to_str(indices):
	s = [ALL_CHARS[idx] for idx in indices]
	return s

def collate_fn(data):
	specgram_list = [sample[0] for sample in data]
	text_list = [sample[1] for sample in data]
	
	spec_lengths = []
	text_lengths = []
	for sample in data:
		specgram = sample["specgram"]
		spec_lengths.append(specgram.size(2))
		text_lengths.append(text.size(0))
	max_spec_len = max(spec_lengths)
	max_text_len = max(text_lengths)

	#TODO: spec/text pad & collate, obtain text mask


def test():
	root_dir = "data/LibriSpeech/dev-clean/"
	dataset = SpeechDataset(root_dir)
	for i in range(len(dataset)):
		sample = dataset[i]
		specgram = sample["specgram"]
		text = sample["text"]
		print(specgram.size(), text.size())
		if i==3:
			break

test()
