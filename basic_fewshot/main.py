# implmentation of early few-shot learning methods

from dataset import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def siamese_expr(train_data, test_data):
	batch_size = 128
	n_pairs = 10000
	C = 5
	K = 3
	train_loader = get_siamese_loader(train_data, n_pairs, batch_size)
	test_loader = get_episode_loader(test_data, C, K)
	net = SiameseNetwork()
	net.to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	for epoch in range(20):
		train_loss = train_epoch(net, train_loader, optimizer)
		test_acc = test_epoch(net, test_loader)
		print("(epoch {}) train loss: {:g} test acc: {:g}".format(epoch, train_loss, test_acc))

def train_epoch(net, train_loader, optimizer):
	net.train()
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

def test_epoch(net, test_loader):
	net.eval()
	running_correct = 0
	running_Q = 0
	with torch.no_grad():
		for batch_i, batch in enumerate(test_loader):
			support = batch["support"].to(device)
			query = batch["query"].to(device)
			label = batch["label"].to(device)
			pred = net.infer(support, query)
			running_Q += query.size(1)
			running_correct += torch.sum(pred == label).item()
	return running_correct/running_Q

def test1():
	train_data, test_data = load_omniglot()
	siamese_expr(train_data, test_data)

test1()
