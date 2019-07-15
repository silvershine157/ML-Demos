import numpy as np
import torchvision
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

data_dir = "./data/flickr8k/"

# original data
caption_file = data_dir + "Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt"
image_dir = data_dir + "Flickr_Data/Images/"

# intermediate data
load_intermediate_data = True
voca_file = data_dir + "processed/voca.txt"
image_names_file = data_dir + "processed/image_names.txt"
captions_file = data_dir + "processed/captions_file"

# CNN activations
load_cnn_activations = True
CNN_BATCH_SIZE = 1024
cnn_activations_file = data_dir + "processed/cnn_activations"

# minimum word count to be kept in voc
# original paper fixes the vocabulary size to 10000
MIN_WORD_COUNT = 3

# maximum caption legnth (does not count <start>, <end>)
MAX_CAPTION_LENGTH = 10

BATCH_SIZE = 4

# resizing
IMG_SIZE = 224 # >= 224

# DEBUG
NUM_LINES = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Vocabulary mapping inspired by pytorch chatbot tutorial
class Voc(object):

	def __init__(self):
		self.word2idx = {}
		self.word2cnt = {}
		self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>"}
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
		self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>"}
		self.num_words = 3
		for word in keep_words:
			self.add_word(word)


def process_caption_file(caption_file, num_lines=None):

	print("Processing caption file . . .")
	
	# read caption file
	with open(caption_file) as f:
		_ = f.readline() # ignore first line
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

		if(len(img_name) == 0 or img_name != img_name[-1]):
			# new image
			orig_image_names.append(img_name)
			orig_captions.append([])
		
		voc.add_words(words) # update vocabulary
		orig_captions[-1].append(words)

	# trim infrequent word
	voc.trim(MIN_WORD_COUNT)

	# discard image if one of the captions are invalid
	# possible alternative is to have different number of captions per image
	survived_img_indices = []
	for idx, caption_group in enumerate(orig_captions):
		survive = True
		for caption in caption_group:
			
			# check length
			if len(caption) > MAX_CAPTION_LENGTH:
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

def make_cnn_activations(image_names, image_dir, cnn_batch_size):

	# inspired by pytorch transfer learning tutorial
	print("Extracting CNN features . . .")
	cnn_activations = None

	# normalizing transforation
	trans = torchvision.transforms.Compose([
		torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	img_dataset = ImageDataset(image_dir, image_names, trans)

	
	# load pretrained CNN as feature extractor
	cnn_model = torchvision.models.resnet18(pretrained=True)
	for param in cnn_model.parameters():
		param.requires_grad = False
	cnn_extractor = torch.nn.Sequential(*list(cnn_model.children())[:-2])
	cnn_extractor = cnn_extractor.to(device)


	dataloader = DataLoader(img_dataset, batch_size=cnn_batch_size, shuffle=False)
	activations_list = []

	for i_batch, batch in enumerate(dataloader):
		print("( %d / %d )"%(i_batch * cnn_batch_size, len(image_names)))
		batch = batch.to(device)
		activation = cnn_extractor(batch)
		activations_list.append(activation.to("cpu"))

	cnn_activations = torch.cat(activations_list, dim=0)
	print("activation shape: "+str(list(cnn_activations.shape)))

	return cnn_activations


def sample_batch(cnn_activations, captions, batch_size):

	batch_idx = np.random.randint(len(captions), size=batch_size)
	activation_batch = cnn_activations[batch_idx]
	caption_batch = []
	for idx in batch_idx:
		caption_group = captions[idx]
		caption_batch.append(random.choice(caption_group))

	batch = (activation_batch, caption_batch)
	return batch


def batch_to_train_data(voc, batch):

	

	return None, None, None, None



# don't write files
def test():

	voc, captions, image_names = process_caption_file(caption_file, num_lines=NUM_LINES)

	'''
	voc: maintains word <-> idx mapping info
	captions: list of N caption groups
		- caption group is a list of captions for an image
		- caption does not include special tokens
	image_names
		- list of image file names, order consistent with captions
	'''

	cnn_activations = make_cnn_activations(image_names, image_dir, CNN_BATCH_SIZE)

	
	'''
	cnn_activations: (N, C, W, H) tensor where
	- N: number of images
	- W: horizontal spatial dimension
	- H: vertical spatial dimension
	- C: numver of output channels (not color)
	'''

	# we now have complete training data in our variables

	batch = sample_batch(cnn_activations, captions, BATCH_SIZE)
	'''
	batch: list of pairs where for each pair,
	first element: (1, C, W, H) activation
	second element: single caption (not caption group)
	'''



	cnn_activation, caption, mask, max_target_len = batch_to_train_data(voc, batch)
	'''
	cnn_activation: (B, C, W, H)
	caption: (max_target_len, B)
	mask: (max_target_len, B)
	max_target_len: integer
	'''


def debug(caption_file):
	with open(caption_file) as f:
		_ = f.readline() # ignore first line
		lines = f.readlines()

	for line in lines:
		# read image name & words
		tokens = line.strip().split()
		img_name = (tokens[0].split("#"))[0]

		print(img_name)

def main():
	# TODO
	pass

#debug(caption_file)
test()





