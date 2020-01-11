import torch
from model import *
from const import *

def test1():

	n_blocks = 6
	d_model = 512
	vsize_src = 100
	vsize_tar = 128
	d_ff = 2048
	net = Transformer(n_blocks, d_model, vsize_src, vsize_tar, d_ff)

	batch_size=4
	len_src=10
	len_tar=8
	source = torch.zeros([batch_size, len_src], dtype=torch.long)
	target = torch.zeros([batch_size, len_tar], dtype=torch.long)
	src_mask = torch.zeros([batch_size, len_src], dtype=torch.bool)
	tar_mask = torch.zeros([batch_size, len_tar], dtype=torch.bool)
	for b in range(batch_size):
		src_mask[b, np.random.randint(len_src//2, len_src):] = 1
		tar_mask[b, np.random.randint(len_tar//2, len_tar):] = 1

	print(source.shape)
	print(target.shape)
	loss = net.loss(source, src_mask, target, tar_mask)
	print(loss)


def test2():
	# masking test
	batch_size=4
	len_src=10
	len_tar=8
	source = torch.zeros([batch_size, len_src], dtype=torch.long)
	target = torch.zeros([batch_size, len_tar], dtype=torch.long)
	src_mask = torch.zeros([batch_size, len_src], dtype=torch.bool)
	tar_mask = torch.zeros([batch_size, len_tar], dtype=torch.bool)
	for b in range(batch_size):
		src_mask[b, np.random.randint(len_src//2, len_src):] = 1
		tar_mask[b, np.random.randint(len_tar//2, len_tar):] = 1
	enc_self = expand_mask(src_mask, Lq=None, autoreg=False)
	dec_self = expand_mask(tar_mask, Lq=None, autoreg=True)
	dec_cross = expand_mask(src_mask, Lq=len_tar, autoreg=False)
	#print(dec_cross)


def test3():
	# decoding test
	n_blocks = 6
	d_model = 512
	vsize_src = 100
	vsize_tar = 5
	d_ff = 2048
	net = Transformer(n_blocks, d_model, vsize_src, vsize_tar, d_ff)
	batch_size=4
	len_src=10
	source = torch.zeros([batch_size, len_src], dtype=torch.long)
	src_mask = torch.zeros([batch_size, len_src], dtype=torch.bool)
	for b in range(batch_size):
		src_mask[b, np.random.randint(len_src//2, len_src):] = 1

	source = source.to(device)
	src_mask = src_mask.to(device)
	net.to(device)

	res = net.decode(source, src_mask)
	pred_batch = res.tolist()
	print(pred_batch)

test3()