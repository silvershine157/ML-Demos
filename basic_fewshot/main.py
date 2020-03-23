# implmentation of early few-shot learning methods

from dataset import *

def siamese_expr(train_data, test_data):
	batch_size = 12
	n_pairs = 10000
	train_loader = get_siamese_loader(train_data, n_pairs, batch_size)
	for batch_i, batch in enumerate(train_loader):
		pair, label = batch
		print(pair.shape)
		print(label.shape)
		break

def matching_expr():
	pass

def prototypical_expr():
	pass

def main():
	train_data, test_data = load_omniglot()
	siamese_expr(train_data, test_data)

main()