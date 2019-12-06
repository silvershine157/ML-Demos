import torch
import torch.nn as nn

def test():
	n = 4
	d = 16
	X = torch.FloatTensor(n, d).uniform_()
	Z = Encoder(X)
	res = Decoder(Z)
	print(res.shape)

# Attention
def Att(Q, K, V):
	'''
	<input>
	Q: n x d_q
	K: n_v x d_q
	V: n_v x d_v
	<output>
	n x d_v
	'''
	scores = torch.matmul(Q, K.t()) # n x n_v
	exp_scores = torch.exp(scores)
	softmax_scores = exp_scores/exp_scores.sum(dim=1, keepdim=True)
	res = torch.matmul(softmax_scores, V) # n x d_v
	return res

# Multi-head attention
def Multihead(Q, K, V):
	'''
	<input>
	Q: n x d
	K: n_v x d
	V: n_v x d
	<output>
	n x d
	'''
	h = 8
	d = Q.size(1)
	d_M = d//h
	W_Q = [torch.FloatTensor(d, d_M).uniform_() for _ in range(h)]
	W_K = [torch.FloatTensor(d, d_M).uniform_() for _ in range(h)]
	W_V = [torch.FloatTensor(d, d_M).uniform_() for _ in range(h)]
	W_O = torch.FloatTensor(d, d).uniform_()
	O = []
	for j in range(h):
		O_j = Att(
			torch.matmul(Q, W_Q[j]),
			torch.matmul(Q, W_K[j]),
			torch.matmul(Q, W_V[j])
		)
		O.append(O_j)
	O_cat = torch.cat(O, dim=1)
	res = torch.matmul(O_cat, W_O)
	return res

# Row-wise feedforward layer
def rFF(H):
	'''
	<input>
	H: n x d
	<output>
	n x d
	'''
	d = H.size(1)
	d_ff = 4 * d
	W1 = torch.FloatTensor(d, d_ff).uniform_()
	b1 = torch.FloatTensor(1, d_ff).uniform_()
	W2 = torch.FloatTensor(d_ff, d).uniform_()
	b2 = torch.FloatTensor(1, d).uniform_()
	relu = nn.ReLU()
	z1 = relu(torch.matmul(H, W1) + b1) # n x d_ff
	z2 = relu(torch.matmul(z1, W2) + b2) # n x d
	return z2

# LayerNorm
def LayerNorm(X):
	'''
	<input>
	X: n x d
	<output>
	n x d
	'''
	# normalize in d
	d = X.size(1)
	layernorm = nn.LayerNorm(d)
	return layernorm(X)

# Multihead Attention Block
def MAB(X, Y):
	'''
	<input>
	X: n x d
	Y: n_y x d
	<output>
	n x d
	'''
	H = LayerNorm(X + Multihead(X, Y, Y))
	return LayerNorm(H + rFF(H))

# Set Attention Block
def SAB(X):
	'''
	<input>
	X: n x d
	<output>
	n x d
	'''
	return MAB(X, X)

# Induced Set Attention Block
def ISAB(X):
	'''
	<input>
	X: n x d
	<output>
	n x d
	'''
	I = None # n x d
	H = MAB(I, X)
	return MAB(X, H)

# Encoder
def Encoder(X):
	'''
	<input>
	X: n x d
	<output>
	n x d
	'''
	# example
	return SAB(SAB(X)) 

# Pooling by Multihead Attention
def PMA(Z):
	'''
	<input>
	Z: n x d
	<output>
	k x d
	'''
	d = Z.size(1)
	k = 5
	S = torch.FloatTensor(k, d).uniform_()  # k x d
	return MAB(S, rFF(Z))

# Decoder
def Decoder(Z):
	'''
	<input>
	Z: n x d
	<output>
	Y: k x d
	'''
	return rFF(SAB(PMA(Z)))


test()