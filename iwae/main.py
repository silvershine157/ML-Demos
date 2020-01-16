import torch
import os
import argparse
import torch.optim as optim

from dataset import *
from model import *
from const import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', type=str)
parser.add_argument('--load_model', default='', type=str)
# hyperparams
parser.add_argument('--latent', default='10', type=int)
# train settings
parser.add_argument('--epochs', default='100', type=int)
parser.add_argument('--batch_size', default='64', type=int)
parser.add_argument('--lr', default='0.001', type=float)
parser.add_argument('--print_every', default='1', type=int)
parser.add_argument('--validate_every', default='10', type=int)
# flags
parser.add_argument('--train', action='store_true')
parser.add_argument('--load_preproc', action='store_true')
parser.add_argument('--download', action='store_true')
args = parser.parse_args()



def train_epoch(net, loader, optimizer):
	running_loss = 0.0
	running_n = 0
	net.to(device)
	for batch_idx, batch in enumerate(loader):
		optimizer.zero_grad()
		image = batch["image"].to(device)
		#loss = net.loss(image)
		#loss = net.loss_alternate(image)
		loss = net.iwae_loss(image)
		loss.backward()
		optimizer.step()
		n = image.size(0)
		running_loss += n * loss.item()
		running_n += n
	avg_loss = running_loss / running_n
	return avg_loss

def test_epoch(net, loader):
	running_loss = 0.0
	running_n = 0
	net.to(device)
	for batch_idx, batch in enumerate(loader):
		image = batch["image"].to(device)
		#loss = net.loss(image)
		#loss = net.loss_alternate(image)
		loss = net.iwae_loss(image)
		n = image.size(0)
		running_loss += n * loss.item()
		running_n += n
	avg_loss = running_loss / running_n	
	return avg_loss

def make_z_grid(Dz, N):
	'''
	Dz: int
	---
	z_grid: [N, N, Dz]
	'''
	if Dz == 2:
		# coordinate grid
		z_grid = torch.zeros(N, N, Dz)
		linsp = torch.linspace(-1, 1, N)
		z_grid[:, :, 0] = linsp.view(-1, 1)
		z_grid[:, :, 1] = linsp.view(1, -1)
	else:
		# sample randomly
		z_grid = torch.randn(N, N, Dz)

	return z_grid


def show_img_grid(net, z_grid):
	'''
	net: VAE
	z_grid: [N, N, Dz]
	'''
	img_grid = None # img_grid: [N, N, H, W]


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

	# setup model & dirs
	if args.load_model != '':
		print("Loading model . . .")
		pass
	else:
		print("Initializing model . . .")
		net = VAE(28,28,args.latent)
	model_root = 'data/models/'
	if not os.path.exists(model_root):
		os.makedirs(model_root)
	i = 0
	while True:
		model_name = args.name + "(%d)"%(i)
		model_dir = os.path.join(model_root, model_name)
		if os.path.exists(model_dir):
			i += 1
			continue
		else:
			os.makedirs(model_dir)
			break

	# train model
	if args.train:
		print("Training model . . .")
		net.train()
		optimizer = optim.Adam(net.parameters(), lr=args.lr)
		best_loss = None
		for epoch in range(1, args.epochs+1):
			loss = train_epoch(net, train_loader, optimizer)
			if epoch % args.print_every == 0:
				print("(Epoch %d) Training loss: %4f"%(epoch, loss))
			if epoch % args.validate_every == 0:
				net.eval()
				loss = test_epoch(net, val_loader)
				print("Validation loss: %4f"%(loss))
				if best_loss==None or loss > best_loss:
					# save best model
					best_loss = loss
					fname = model_name+'_best_(epoch%d)'%(epoch)
					torch.save(net, os.path.join(model_dir, fname))
				net.train()

	# test model
	print("Testing model . . .")
	net.eval()
	loss = test_epoch(net, test_loader)
	print("Test loss: %4f"%(loss))
	pass


if __name__ == "__main__":
	main()

