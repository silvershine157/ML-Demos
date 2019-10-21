import numpy as np
from scipy.optimize import linear_sum_assignment

def match_misassigned_labels(z_true, z_pred):
	'''
	params:
		z_true: N ground truth labels from {0, ..., K-1}
		z_pred: N predicted labels with unknown permutation
	returns:
		accuracy: maximum accuracy achieved with the right assignment
		perm: corresponding assignment from z_true to z_pred
	'''
	N = z_true.size
	K = max(np.max(z_true), np.max(z_pred))+1
	match_counts = np.zeros((K, K), dtype=np.uint32)
	for n in range(N):
		match_counts[z_true[n], z_pred[n]] += 1
	cost = 1.0-match_counts/N # seems to work only for nonnegative weights
	row_ind, col_ind = linear_sum_assignment(cost)
	accuracy = np.sum(match_counts[row_ind, col_ind])/N
	perm = col_ind # row_ind is (0, ..., K-1)

	return accuracy, perm

def test():
	N = 1000 # number of datapoints
	K = 10 # number of label classes
	alpha = 1.0 # uniform Dirichlet parameter
	corrupt_ratio = 0.1

	# generate ground truth labels
	distr = np.random.dirichlet(alpha * np.ones(K))
	z_true = np.random.choice(K, size=(N), p=distr)

	# corrupt & permute
	noise = np.random.randint(K, size=(N))
	noise_mask = np.random.choice(2, size=(N), p=[1-corrupt_ratio, corrupt_ratio])
	z_corrupt = np.mod(z_true + noise_mask*noise, K)
	true_perm = np.random.permutation(K)
	z_pred = true_perm[z_corrupt]

	accuracy, est_perm = match_misassigned_labels(z_true, z_pred)
	print("accuracy for best asisgnment: %04f"%(accuracy))
	print("true permutation:\t", true_perm)
	print("estimated permutation:\t", est_perm)
