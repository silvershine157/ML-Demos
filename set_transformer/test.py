import torch

from models import SetTransformer

def test():
	batch_size = 4
	input_size = 7
	dim_input = 8
	num_outputs = 2
	dim_output = 3
	net = SetTransformer(dim_input, num_outputs, dim_output)
	X = torch.FloatTensor(batch_size, input_size, dim_input).uniform_()
	out = net(X)
	print(out.shape)

test()