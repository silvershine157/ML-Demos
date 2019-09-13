from dataset import *

def test1():
	data = preprocess_data()
	print(type(data["train_img"]))
	print(data["train_img"].shape)
	visualize_img(data["train_img"][100, :, :])
	pass

def visualize_img(img):
	for r in range(28):
		row = ''
		for c in range(28):
			if img[r][c] > 100:
				row = row + "# "
			else:
				row = row + ". "
		print(row)

test1()
