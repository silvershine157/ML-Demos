import torch
import math

def SDPA(Q, K, V):
	'''
	Q: L_q x d_k
	K: L_v x d_k
	V: L_v x d_v
	out: L_q x d_v
	'''
	d_k = Q.size(1)
	softmax = torch.nn.Softmax(dim=1)
	scores = torch.matmul(Q, K.t()) # TODO: softamx
	weight_mat = softmax(scores / math.sqrt(d_k))

	return torch.matmul(weight_mat, V)

def MHA(Q, K, V, W_q, W_k, W_v, W_o):
	'''
	Q: L_q x d_m
	K: L_v x d_m
	V: L_v x d_m
	W_q: d_m x d_k*h
	W_k: d_m x d_k*h
	W_v: d_m x d_v*h
	W_o: d_v*h x d_m
	'''

def test_sdpa():
	L_q = 10
	L_v = 15
	d_k = 5
	d_v = 7
	Q = torch.rand((L_q, d_k))
	K = torch.rand((L_v, d_k))
	V = torch.rand((L_v, d_v))
	out = SDPA(Q, K, V)
	print(out.shape)


def test_mha():
	L_q = 10
	L_v = 15
	d_m = 24
	h = 8
	d_k = d_m//h
	d_v = d_m//h
	Q = torch.rand((L_q, d_m))
	K = torch.rand((L_v, d_m))
	V = torch.rand((L_v, d_m))
	W_qs = [torch.rand((d_m, d_k)) for _ in range (h)]
	W_ks = [torch.rand((d_m, d_k)) for _ in range (h)]
	W_vs = [torch.rand((d_m, d_v)) for _ in range (h)]
	W_o = torch.rand((h*d_v, d_m))

	# fist method
	heads = []
	for i in range(h):
		heads.append(SDPA(
			torch.matmul(Q, W_qs[i]),
			torch.matmul(K, W_ks[i]),
			torch.matmul(V, W_vs[i])
		))
	head_cat = torch.cat(heads, dim=1)
	out1 = torch.matmul(head_cat, W_o)

	# second method
	W_os = torch.split(W_o, d_v, dim=0)
	W_us = []
	for i in range(h):
		W_us.append(torch.matmul(W_vs[i], W_os[i]))
	W_u_cat = torch.cat(W_us, dim=0)

	heads_noWv = []
	for i in range(h):
		heads_noWv.append(SDPA(
			torch.matmul(Q, W_qs[i]),
			torch.matmul(K, W_ks[i]),
			V
		))
	head_cat_noWv = torch.cat(heads_noWv, dim=1)
	out2 = torch.matmul(head_cat_noWv, W_u_cat)

	# test
	print(out1.shape)
	print(out2.shape)
	print(torch.mean(torch.abs(out1 - out2)))
	print(torch.mean(torch.abs(out1)+torch.abs(out2)))

	# verify parameter reduction
	print(d_m*d_v*h + d_v*h*d_m)
	print(d_m*h*d_m)

test_mha()