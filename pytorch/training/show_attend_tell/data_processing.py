import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch import optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import itertools
from constants import device, PAD_token, START_token, END_token


# Voc: Vocabulary mapping class inspired by pytorch chatbot tutorial

class Voc(object):

	def __init__(self):
		self.word2idx = {}
		self.word2cnt = {}
		self.idx2word = {PAD_token: "<pad>", START_token: "<start>", END_token: "<end>"}
		self.num_words = 3
		self.trimmed = False

	def add_words(self, words):
		for word in words:
			self.add_word(word.lower())

	def add_word(self, word):
		if word not in self.word2idx:
			self.word2idx[word] = self.num_words
			self.word2cnt[word] = 1
			self.idx2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2cnt[word] += 1

	def trim(self, min_count):

		# trim only once
		if self.trimmed:
			return
		self.trimmed = True

		keep_words = []
		for word, cnt in self.word2cnt.items():
			if cnt >= min_count:
				keep_words.append(word)
		print("Keep %d words among %d words (not counting special tokens)"%(len(keep_words), len(self.word2idx)))

		# give new indices
		self.word2idx = {}
		self.word2cnt = {}
		self.idx2word = {PAD_token: "<pad>", START_token: "<start>", END_token: "<end>"}
		self.num_words = 3
		for word in keep_words:
			self.add_word(word)



## Preprocessing caption

'''
<Output>
voc: maintains word <-> idx mapping info
captions: list of N caption groups
	- caption group is a list of captions for an image
	- caption does not include special tokens
image_names
	- list of image file names, order consistent with captions
'''

def process_caption_file(caption_file, min_word_count, max_caption_length, num_lines=None):

	print("Processing caption file . . .")
	
	# read caption file
	with open(caption_file) as f:
		lines = f.readlines()
		# mini data for debugging
		if(num_lines):
			lines = lines[:num_lines]

	voc = Voc() # maintains word - idx mapping
	orig_image_names = []
	orig_captions = []
	for line in lines:
		# read image name & words
		tokens = line.strip().split()
		img_name = (tokens[0].split("#"))[0]

		# Skip lines with errorneous image names such as: 2258277193_586949ec62.jpg.1
		if(img_name[-4:] != ".jpg"):
			#print(img_name)
			continue

		words = tokens[1:]
		words = [word.lower() for word in words]
		if(len(orig_image_names) == 0 or img_name != orig_image_names[-1]):
			# new image
			orig_image_names.append(img_name)
			orig_captions.append([])
		
		voc.add_words(words) # update vocabulary
		orig_captions[-1].append(words)

	# trim infrequent word
	voc.trim(min_word_count)

	# discard image if one of the captions are invalid
	# possible alternative is to have different number of captions per image
	survived_img_indices = []
	for idx, caption_group in enumerate(orig_captions):
		survive = True
		for caption in caption_group:
			# check length
			if len(caption) > max_caption_length:
				survive = False
				break
			# check vocabulary
			for word in caption:
				if word not in voc.word2idx:
					survive = False
					break
			if not survive:
				break
		if survive:
			survived_img_indices.append(idx)

	captions = [orig_captions[idx] for idx in survived_img_indices]
	image_names = [orig_image_names[idx] for idx in survived_img_indices]

	print("Keep %d images among %d images"%(len(image_names), len(orig_image_names)))


	return voc, captions, image_names

def sentence_to_indices(voc, sentence):
	return [voc.word2idx[word] for word in sentence] + [END_token]

def zero_padding(indices_batch, fillvalue=PAD_token):
	# implicitly transpose
	return list(itertools.zip_longest(*indices_batch, fillvalue=fillvalue))

def obtain_mask(indices_batch, value=PAD_token):
	mask = []
	for i, seq in enumerate(indices_batch):
		mask.append([])
		for token in seq:
			if token == PAD_token:
				mask[i].append(0)
			else:
				mask[i].append(1)
	return mask

def caption_list_to_tensor(voc, caption_list):
	# also inspired by pytorch chatbot tutorial
	indices_batch = [sentence_to_indices(voc, sentence) for sentence in caption_list]
	max_target_len = max([len(indices) for indices in indices_batch])
	
	# (max_target_len, B) <- implicitly transposed
	indices_batch = zero_padding(indices_batch)
	mask_batch = obtain_mask(indices_batch)
	mask_batch = torch.ByteTensor(mask_batch)
	indices_batch = torch.LongTensor(indices_batch)

	return indices_batch, mask_batch, max_target_len


def tokens_to_str(tokens, voc):
	tokens = tokens.cpu().numpy()
	L = []
	for token in tokens:
		if token == END_token:
			break
		L.append(voc.idx2word[token])
	s = ' '.join(L)
	return s


## Preprocessing image (getting annotation vectors)

class ImageDataset(Dataset):

	def __init__(self, image_dir, image_names, transform):
		self.image_dir = image_dir
		self.image_names = image_names
		self.transform = transform

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		image_name = self.image_names[idx]
		with Image.open(self.image_dir + image_name) as pil_img:
			img = self.transform(pil_img)
		return img


'''
<Output>
cnn_activations: (N, D, W, H) tensor where
- N: number of images
- W: horizontal spatial dimension
- H: vertical spatial dimension
- D: feature dimension (not color)
'''

def make_cnn_activations(image_names, image_dir, cnn_batch_size=256):
	# inspired by pytorch transfer learning tutorial

	single = (len(image_names)==1)

	if not single:
		print("Extracting CNN features . . .")
	cnn_activations = None

	# normalizing transforation
	trans = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224, 224)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	img_dataset = ImageDataset(image_dir, image_names, trans)
	
	# load pretrained CNN as feature extractor
	cnn_model = torchvision.models.vgg16(pretrained=True)
	for param in cnn_model.parameters():
		param.requires_grad = False
	
	#cnn_extractor = torch.nn.Sequential(*list(cnn_model.children())[:-2]) # resnet18 512 x 7 x 7
	cnn_extractor = torch.nn.Sequential(*list(list(cnn_model.children())[0].children())[:29]) # vgg16 512 x 14 x 14

	if not single:
		cnn_extractor = cnn_extractor.to(device)

	dataloader = DataLoader(img_dataset, batch_size=cnn_batch_size, shuffle=False)
	activations_list = []

	for i_batch, batch in enumerate(dataloader):
		if not single:
			print("( %d / %d )"%(i_batch * cnn_batch_size, len(image_names)))
			batch = batch.to(device)
		activation = cnn_extractor(batch)
		if not single:
			activation = activation.to("cpu")
		activations_list.append(activation)

	cnn_activations = torch.cat(activations_list, dim=0)
	if not single:
		print("activation shape: "+str(list(cnn_activations.shape)))

	return cnn_activations

def flat_annotations(cnn_activations):
	_, a_dim, a_W, a_H = list(cnn_activations.shape)
	a_size = a_W * a_H
	annotations = cnn_activations.view((-1, a_dim, a_size))
	return annotations


'''
<Output>
annotation_batch: (B, D, W, H)
caption_batch: (max_target_len, B)
mask_batch: (max_target_len, B)
max_target_len: integer
'''

def sample_batch(cnn_activations, captions, voc, batch_size):

	batch_idx = np.random.randint(len(captions), size=batch_size)
	annotation_batch = flat_annotations(cnn_activations[batch_idx])
	caption_batch_list = []
	for idx in batch_idx:
		caption_group = captions[idx]
		caption_batch_list.append(random.choice(caption_group))

	indices_batch, mask_batch, max_target_len = caption_list_to_tensor(voc, caption_batch_list)

	return annotation_batch, indices_batch, mask_batch, max_target_len


