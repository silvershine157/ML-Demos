from dataset import *
from model import *

def test1():
	data = preprocess_data()
	print(data["train_images"].dtype)
	print(data["train_images"].shape)
	print(data["train_labels"].dtype)
	print(data["train_labels"].shape)
	pass

def test2():
	data = preprocess_data()
	train_loader, val_loader, test_loader = get_dataloader(data, 4)
	net = Net()
	for b_i, batch in enumerate(train_loader):
		net(batch["image"])
		break

test2()
