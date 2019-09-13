from dataset import *

def test1():
	data = preprocess_data()
	print(data["train_images"].dtype)
	print(data["train_images"].shape)
	print(data["train_labels"].dtype)
	print(data["train_labels"].shape)
	pass

test1()
