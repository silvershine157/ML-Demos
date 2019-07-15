import numpy as np
import normalizer

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
MIN_COUNT = 3

# maximum caption legnth, longer captions are discarded
MAX_LENGTH = 10

BATCH_SIZE = 4

def sampleBatch():
	# TODO
	pass

def batch2TrainData(voc, batch):
	# TODO
	return cnn_activation, caption, mask, max_target_len


# don't write files
def test():

	voc, captions, image_names = process_captions(caption_file)
	'''
	voc: maintains word <-> idx mapping info
	captions: list of N caption groups
		- caption group is a list of captions for an image
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

	batch = sampleBatch(cnn_activations, captions, BATCH_SIZE)
	'''
	batch: list of pairs where for each pair,
	first element: (1, W, H, C) activation
	second element: single caption (not caption group)
	'''

	cnn_activation, caption, mask, max_target_len = batch2TrainData(voc, batch)
	'''
	cnn_activation: (B, W, H, C)
	caption: (max_target_len, B)
	mask: (max_target_len, B)
	max_target_len: integer
	'''

	pass


def main():
	# TODO
	pass

test()