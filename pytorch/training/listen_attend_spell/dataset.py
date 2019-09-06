import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

N_MELS=40

class SpeechDataset(Dataset):
	def __init__(self, root_dir):
		self.metadata = self.scan_metadata(root_dir)

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		audio_filename, text = self.metadata[idx]
		audio_wave, fs = torchaudio.load(audio_filename)
		specgram = torchaudio.transforms.MelSpectrogram(n_mels=N_MELS)(audio_wave)
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
	
	# get data & dimensions
	spec_list = []
	text_list = []
	spec_lengths = []
	text_lengths = []
	for sample in data:
		specgram, text = sample["specgram"], sample["text"]
		spec_lengths.append(specgram.size(2))
		text_lengths.append(text.size(0))
		spec_list.append(specgram)
		text_list.append(text)
	max_spec_len = max(spec_lengths)
	max_text_len = max(text_lengths)

	# pad specgram
	spec_list = [F.pad(spec, (0, max_spec_len-spec.size(2)), mode='constant', value=0) for spec in spec_list]
	specgram_batch = torch.cat(spec_list, dim=0)

	# pad text
	text_list = [F.pad(text, (0, max_text_len-text.size(0)), mode='constant', value=0) for text in text_list]
	text_list = [text.unsqueeze(dim=0) for text in text_list]
	text_batch = torch.cat(text_list, dim=0)
	text_mask = generate_mask(text_batch)

	# construct batch dict
	spec_lengths = torch.ShortTensor(spec_lengths)
	batch = {"specgram":specgram_batch, "text":text_batch, "spec_lengths":spec_lengths, "text_mask":text_mask}
	return batch

def generate_mask(text_batch):
	not_pad = (text_batch != PAD_token)
	return not_pad

def get_data_loader(root_dir, batch_size, shuffle=True):
	dataset = SpeechDataset(root_dir)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
	return loader

def test():
	root_dir = "data/LibriSpeech/dev-clean/"
	loader = get_data_loader(root_dir, batch_size=4, shuffle=True)
	for i, batch in enumerate(loader):
		print(batch["specgram"].size())
		print(batch["text"].size())
		print(batch["text_mask"].size())
		print(batch["spec_lengths"].size())
		break

