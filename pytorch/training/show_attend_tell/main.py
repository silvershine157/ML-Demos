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
from data_processing import Voc, process_caption_file, sample_batch, make_cnn_activations, flat_annotations, tokens_to_str 
from constants import device, PAD_token, START_token, END_token
from config import *


# split to train / validation / test set
def split_dataset(bundle, val_ratio=0.1, test_ratio=0.1):
	
	captions, cnn_activations = bundle
	N = len(captions)
	N_val = int(N * val_ratio)
	N_test = int(N * test_ratio)
	N_train = N - N_val - N_test

	train_bundle = (captions[:N_train], cnn_activations[:N_train])
	val_bundle = (captions[N_train:-N_test], cnn_activations[N_train:-N_test])
	test_bundle = (captions[-N_test:], cnn_activations[-N_test:])
	
	return train_bundle, val_bundle, test_bundle


# training loop
def train_model(model, voc, train_bundle, val_bundle, n_iteration, learning_rate, clip, model_save_file):

	print("Training model . . .")

	# build optimizers
	opt = optim.Adam(model.parameters(), lr=learning_rate)

	# load batches for each iteration (allowing multiple captions for single image)
	n_train_batches = 1000
	n_val_batches = 100 # TODO: auto sizing
	training_batches = [sample_batch(train_bundle, voc, BATCH_SIZE) for _ in range(n_train_batches)]
	val_batches = [sample_batch(val_bundle, voc, BATCH_SIZE) for _ in range(n_val_batches)]

	print_loss = 0.0
	for iteration in range(n_iteration):
		opt.zero_grad()
		loss, norm_loss = model(training_batches[iteration%n_train_batches])
		loss.backward()
		_ = nn.utils.clip_grad_norm_(model.parameters(), clip)
		opt.step()
		print_loss += norm_loss
		if iteration % PRINT_EVERY == PRINT_EVERY - 1:
			print("Training iterations: %d, loss: %f"%(iteration+1, print_loss / PRINT_EVERY))
			print_loss = 0.0
		if iteration % SAVE_EVERY == SAVE_EVERY - 1:
			# validation
			with torch.no_grad():
				val_loss = 0.0
				for val_batch in val_batches:
					_, norm_loss = model(val_batch)
					val_loss += norm_loss
				torch.save(model.state_dict(), model_save_file + "_%07d"%(iteration+1))
				print("Validation loss: %f, model saved."%(val_loss / n_val_batches))

	print("Training complete!")
	return


# interactive evaluation: read image, produce caption & BLEU score
def interactive_test(model, image_dir, voc, all_captions, all_image_names):

	print("Interactive test")

	chencherry = SmoothingFunction()
	
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
		s = tokens_to_str(all_tokens[0], voc)
		print(s)
		
		# report BLEU
		idx = all_image_names.index(q)
		bleu_score = sentence_bleu(all_captions[idx], s.split(), smoothing_function=chencherry.method1)
		print("BLEU: %.4f"%bleu_score)


# batch testing on test datset
def report_bleu(model, bundle, voc):
	
	print("Calculating average BLEU score . . .")

	captions, cnn_activations = bundle
	chencherry = SmoothingFunction()
	
	N = len(captions)
	bleu_sum = 0.0
	for n in range(N):
		if n%100 == 0:
			print("(%d / %d)"%(n, N))
		ref = captions[n]
		annotation = flat_annotations(cnn_activations[n].unsqueeze(dim=0))
		all_tokens = model.greedy_decoder(annotation, MAX_CAPTION_LENGTH+2)
		s = tokens_to_str(all_tokens[0], voc)
		bleu = sentence_bleu(captions[n], s.split(), smoothing_function=chencherry.method1)
		bleu_sum += bleu
	print("Average BLEU score: %.4f"%(bleu_sum/N))


# main routine
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

	# split dataset
	bundle = (captions, cnn_activations)
	train_bundle, val_bundle, test_bundle = split_dataset(bundle)

	# setup model
	model = SoftSATModel(cnn_activations.shape, voc.num_words, EMBEDDING_DIM, CELL_DIM)
	if LOAD_MODEL:
		model.load_state_dict(torch.load(MODEL_LOAD_FILE))
	model = model.to(device)

	# train model
	if TRAIN:
		model.train() # train mode
		train_model(model, voc, train_bundle, val_bundle, N_ITERATIONS, LEARNING_RATE, CLIP, MODEL_SAVE_FILE)

	# generate caption for given image filename
	model.eval() # evaluation mode
	if BATCH_TEST:
		report_bleu(model, test_bundle, voc)
	else:
		interactive_test(model, PATHS["orig_image_dir"], voc, captions, image_names)


main()



