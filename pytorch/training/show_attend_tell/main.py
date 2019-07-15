import numpy as np
from normalizer import Voc

data_dir = "./data/flickr8k/"

# original data
caption_file = data_dir + "Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt"
image_dir = data_dir + "Flickr_Data/Images"

# intermediate data
load_intermediate_data = True
voca_file = data_dir + "processed/voca.txt"
image_names_file = data_dir + "processed/image_names.txt"
captions_file = data_dir + "processed/captions_file"

# CNN activations
load_cnn_activations = True
cnn_activations_file = data_dir + "processed/cnn_activations"

# minimum word count to be kept in voc
# original paper fixes the vocabulary size to 10000
MIN_WORD_COUNT = 20

# maximum caption legnth (does not count <start>, <end>)
MAX_CAPTION_LENGTH = 10

BATCH_SIZE = 4

# DEBUG
NUM_LINES = None


def process_caption_file(caption_file, num_lines=None):

	print("Processing caption file . . .")
	
	with open(caption_file) as f:
		# ignore first line
		_ = f.readline()
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



def make_cnn_activations(image_names, image_dir):
	return None

def sample_batch(cnn_activations, captions, batch_size):
	return None

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

	cnn_activations = make_cnn_activations(image_names, image_dir)
	'''
	cnn_activations: (N, W, H, C) tensor where
	- N: number of images
	- W: horizontal spatial dimension
	- H: vertical spatial dimension
	- C: numver of output channels (not color)
	'''

	# we now have complete training data in our variables

	batch = sample_batch(cnn_activations, captions, BATCH_SIZE)
	'''
	batch: list of pairs where for each pair,
	first element: (1, W, H, C) activation
	second element: single caption (not caption group)
	'''

	cnn_activation, caption, mask, max_target_len = batch_to_train_data(voc, batch)
	'''
	cnn_activation: (B, W, H, C)
	caption: (max_target_len, B)
	mask: (max_target_len, B)
	max_target_len: integer
	'''



def main():
	# TODO
	pass

test()