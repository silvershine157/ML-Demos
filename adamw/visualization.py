import numpy as np
import matplotlib.pyplot as plt

# parse and plot data
def parse_data(filename):
	data = []
	with open(filename, 'r') as f:
		for l in f:
			L = l.strip().split()
			if(len(L) > 0 and L[0] == 'Train'):
				data.append([float(L[2]), float(L[5])])
	return np.array(data)

def vis1():
	# 1. SGD+l2reg best train/test LC
	# 2. Adam best train/test LC
	# 3. AdamW best train/test LC
	# all in one graph
	pass

def test1():
	data1 = parse_data("data/exp_results/sgd_lr_e-2_l2reg_e-2.txt")
	plt.plot(data1[:, 0], 'c-', label='SGD train')
	plt.plot(data1[:, 1], 'c--', label='SGD test')
	data2 = parse_data("data/exp_results/adam_lr_e-4_l2reg_e-2.txt")
	plt.plot(data2[:, 0], 'b-', label='Adam train')
	plt.plot(data2[:, 1], 'b--', label='Adam test')
	data3 = parse_data("data/exp_results/adamW_lr_e-4_wd_e0.txt")
	plt.plot(data3[:, 0], 'm-', label='AdamW train')
	plt.plot(data3[:, 1], 'm--', label='AdamW test')
	plt.legend()
	plt.show()

test1()