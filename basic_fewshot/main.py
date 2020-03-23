# implmentation of early few-shot learning methods

from dataset import *
from model import *

def siamese_expr(train_data, test_data):
	batch_size = 12
	n_pairs = 10000
	train_loader = get_siamese_loader(train_data, n_pairs, batch_size)
	net = SiameseNetwork()
	loss = train_epoch(train_loader, net)

def train_epoch(train_loader, net):
	for batch_i, batch in enumerate(train_loader):
		pair, label = batch
		loss = net.loss(pair, label)
		break

def main():
	train_data, test_data = load_omniglot()
	siamese_expr(train_data, test_data)


main()
