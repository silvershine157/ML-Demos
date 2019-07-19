'''
References:
PyTorch chatbot tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
PyTorch transfer learning tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch import optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import itertools
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# import modules
from model import SoftSATModel 
from data_processing import Voc, process_caption_file, sample_batch, make_cnn_activations, flat_annotations,
tokens_to_str 
from constants import device, PAD_token, START_token, END_token
from config import *

# training iterations
def train_model(model, voc, cnn_activations, captions, n_iteration, learning_rate, clip, model_save_file):

	print("Training model . . .")
	print_every = 100
	save_every = 1000

	# set to train mode
	model.train()

	# build optimizers
	opt = optim.Adam(model.parameters(), lr=learning_rate)

	# load batches for each iteration (allowing multiple captions for single image)
	num_batches = 10000
	training_batches = [sample_batch(cnn_activations, captions, voc, BATCH_SIZE) for _ in range(num_batches)]

	print_loss = 0.0
	for iteration in range(n_iteration):
		opt.zero_grad()
		loss, norm_loss = model(training_batches[iteration%num_batches])
		loss.backward()
		_ = nn.utils.clip_grad_norm_(model.parameters(), clip)
		opt.step()
		print_loss += norm_loss
		if iteration % print_every == print_every - 1:
			print("Iteration: %d, loss: %f"%(iteration+1, print_loss / print_every))
			print_loss = 0.0
		if iteration % save_every == save_every - 1:
			torch.save(model.state_dict(), model_save_file + "_%07d"%(iteration+1))

	print("Training complete!")
	return


# evaluation
def interactive_test(model, image_dir, voc, all_captions, all_image_names):

	while True:
		q = input("image name:")
		if q == 'quit':
			return
		if q not in all_image_names:
			print("wrong name!")
			continue

		# make annotation for image q
		cnn_activations = make_cnn_activations([q], image_dir, 1)
		annotation = flat_annotations(cnn_activations)

		# perform greedy decoding
		all_tokens = model.greedy_decoder(annotation, MAX_CAPTION_LENGTH+2)

		# print result
		s = tokens_to_str(all_tokens[0], voc)
		print(s)
		bleu_score = get_bleu(s.split(), q, all_captions, all_image_names)
		print("BLEU: %.4f"%bleu_score)


def get_bleu(test_caption, test_image_name, all_captions, all_image_names):
	idx = all_image_names.index(test_image_name)
	ref_captions = all_captions[idx]
	chencherry = SmoothingFunction()
	bleu_score = sentence_bleu(ref_captions, test_caption, smoothing_function=chencherry.method1)
	return bleu_score


def main():

	# get voc, captions, image_names
	if NEW_INTERMEDIATE_DATA:
		voc, captions, image_names = process_caption_file(PATHS["orig_caption_file"], MIN_WORD_COUNT, MAX_CAPTION_LENGTH, num_lines=NUM_LINES)
		torch.save((voc, captions, image_names), PATHS["intermediate_data_file"])
	else:
		voc, captions, image_names = torch.load(PATHS["intermediate_data_file"])

	# get cnn activations
	if NEW_CNN_ACTIVATIONS:
		cnn_activations = make_cnn_activations(image_names, PATHS["orig_image_dir"])
		torch.save(cnn_activations, PATHS["cnn_activations_file"])
	else:
		cnn_activations = torch.load(PATHS["cnn_activations_file"])

	# setup model
	model = SoftSATModel(cnn_activations.shape, voc.num_words, EMBEDDING_DIM, CELL_DIM)
	if LOAD_MODEL:
		model.load_state_dict(torch.load(MODEL_LOAD_FILE))
	model = model.to(device)

	# train model
	if TRAIN:
		train_model(model, voc, cnn_activations, captions, N_ITERATIONS, LEARNING_RATE, CLIP, MODEL_SAVE_FILE)

	# generate caption for given image filename
	interactive_test(model, PATHS["orig_image_dir"], voc, captions, image_names)


main()



