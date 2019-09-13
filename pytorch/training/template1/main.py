import os
import argparse
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', type=str)
parser.add_argument('--epochs', default='10', type=int)
parser.add_argument('--train', action='store_true')
parser.add_argument('--load_preproc', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--download' action='store_true')

args = parser.parse_args()

def main():

	# prepare data
	if args.download:
		print("Downloading data . . .")
		download_data()
	if args.load_preproc:
		print("Loading preprocessed data . . .")
	else:
		print("Preprocessing data . . .")
		preprocess_data()
		print("Saving preprocessed data . . .")

	# setup dataloader
	print("Making DataLoader . . .")

	# setup model
	if args.load_model:
		print("Loading model . . .")
		pass
	else:
		print("Initializing model . . .")
		pass

	# train model
	if args.train:
		print("Training model . . .")
		for epoch in range(args.epochs):
			pass

	# test model
	print("Testing model . . .")
	pass


if __name__ == "__main__":
	main()

