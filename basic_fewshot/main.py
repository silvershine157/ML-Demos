# implmentation of early few-shot learning methods

from dataset import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def siamese_expr(train_data, test_data):
	batch_size = 64
	n_pairs = 10000
	train_loader = get_siamese_loader(train_data, n_pairs, batch_size)
	net = SiameseNetwork()
	net.to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	for epoch in range(10):
		loss = train_epoch(net, train_loader, optimizer)
		print("(epoch {}) training loss {:g}".format(epoch, loss))

def train_epoch(net, train_loader, optimizer):
	running_n = 0
	running_loss = 0.0
	for batch_i, batch in enumerate(train_loader):
		optimizer.zero_grad()
		pair, label = batch
		B = pair.size(0)
		loss = net.loss(pair.to(device), label.to(device))
		loss.backward()
		optimizer.step()
		running_loss += B*loss.item()
		running_n += B
	return running_loss/running_n

def main():
	train_data, test_data = load_omniglot()
	siamese_expr(train_data, test_data)


main()
