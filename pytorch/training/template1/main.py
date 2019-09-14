import torch
import os
import argparse
from dataset import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', type=str)
# train settings
parser.add_argument('--epochs', default='10', type=int)
parser.add_argument('--batch_size', default='4', type=int)
parser.add_argument('--print_every', default='5', type=int)
parser.add_argument('--validate_every', default='5', type=int)

# flags
parser.add_argument('--train', action='store_true')
parser.add_argument('--load_preproc', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--download', action='store_true')

args = parser.parse_args()

def mse_loss(y_h, target):
	err = y_h - target
	return torch.mean(torch.mul(err, err))

def train_epoch(net, loader):
	running_loss = 0.0
	running_n = 0
	for batch_idx, batch in enumerate(loader):
		n = batch["image"].size(0)
		y_h = net(batch["image"])
		target = batch["label"]
		batch_loss = mse_loss(y_h, target)
		running_loss += n * batch_loss.item()
		running_n += n
	avg_loss = running_loss / running_n
	return avg_loss

def test_epoch(net, loader):
	running_loss = 0.0
	running_n = 0
	running_correct = 0
	for batch_idx, batch in enumerate(loader):
		n = batch["image"].size(0)
		y_h = net(batch["image"])
		target = batch["label"]
		batch_loss = mse_loss(y_h, target)
		running_loss += n * batch_loss.item()
		label_pred = y_h.argmax(dim=1)
		label_true = target.argmax(dim=1)
		n_correct = (label_pred == label_true).sum()
		running_correct += n_correct.item()
		running_n += n
	avg_loss = running_loss / running_n	
	avg_perf = running_correct / running_n
	return avg_loss, avg_perf


def main():

	# prepare data
	if args.download:
		print("Downloading data . . .")
		download_data()
	if args.load_preproc:
		print("Loading preprocessed data . . .")
		data = torch.load('data/preproc_data')
	else:
		print("Preprocessing data . . .")
		data = preprocess_data()
		print("Saving preprocessed data . . .")
		torch.save(data, 'data/preproc_data')
	print("Making DataLoader . . .")
	train_loader, val_loader, test_loader = get_dataloader(data, args.batch_size)

	# setup model
	if args.load_model:
		print("Loading model . . .")
		pass
	else:
		print("Initializing model . . .")
		net = Net()

	# train model
	if args.train:
		print("Training model . . .")
		net.train()
		best_perf = None
		for epoch in range(1, args.epochs+1):
			loss = train_epoch(net, train_loader)
			if epoch % args.print_every == 0:
				print("(Epoch %d) Training loss: %4f"%(epoch, loss))
			if epoch % args.validate_every == 0:
				loss, perf = test_epoch(net, val_loader)
				print("Validation loss: %4f, accuracy: %g"%(loss, perf))
				if best_perf==None or perf > best_perf:
					# new best model
					best_perf = perf
					torch.save(net, 'data/'+args.name+'_best_(epoch%d)'%(epoch))

	# test model
	print("Testing model . . .")
	net.eval()
	loss, perf = test_epoch(net, test_loader)
	print("Test loss: %4f, accuracy: %g"%(loss, perf))
	pass


if __name__ == "__main__":
	main()

