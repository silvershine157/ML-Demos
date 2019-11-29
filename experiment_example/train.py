import torch
import torch.optim as optim

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mse_loss(y_h, y):
	err = y_h - y
	return torch.mean(torch.mul(err, err))

def train_model(net, train_loader, val_loader, expr=None):
	
	lr=0.001
	epochs=2000
	print_every=500
	validate_every=500
	if expr:
		lmbda=expr.args["lmbda"]

	log("Training model", expr)
	net.train()
	optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=lmbda)
	for epoch in range(1, epochs+1):
		train_loss = train_epoch(net, train_loader, optimizer)
		if epoch % print_every == 0:
			log("(Epoch {0}) Training loss: {1}".format(epoch, train_loss), expr)
			# TODO: use averaged loss
		if epoch % validate_every == 0:
			val_loss = test_epoch(net, val_loader)
			log("Validation loss: {1}".format(epoch, val_loss), expr)

	train_info = None
	return train_info

def test_model(net, test_loader, expr=None):

	log("Training model", expr)
	net.eval()
	test_loss = test_epoch(net, test_loader)
	log("test loss: {0}".format(test_loss), expr)

	test_info = {}
	test_info["test_perf"] = test_loss
	return test_info


def train_epoch(net, loader, optimizer):
	running_loss = 0.0
	running_n = 0
	net.to(device)
	for batch_idx, batch in enumerate(loader):
		optimizer.zero_grad()
		x = batch["x"].to(device)
		y = batch["y"].to(device)
		y_h = net(x)
		batch_loss = mse_loss(y_h, y)
		batch_loss.backward()
		optimizer.step()
		n = x.size(0)
		running_loss += n * batch_loss.item()
		running_n += n
	avg_loss = running_loss / running_n
	return avg_loss

def test_epoch(net, loader):
	running_loss = 0.0
	running_n = 0
	running_correct = 0
	net.to(device)
	for batch_idx, batch in enumerate(loader):
		x = batch["x"].to(device)
		y = batch["y"].to(device)
		y_h = net(x)
		batch_loss = mse_loss(y_h, y)
		n = x.size(0)
		running_loss += n * batch_loss.item()
		running_n += n
	avg_loss = running_loss / running_n
	return avg_loss
