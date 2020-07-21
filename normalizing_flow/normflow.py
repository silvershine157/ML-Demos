import torch
import torch.nn as nn

'''
<convention>
f: x -> z
f_inv: z -> x
'''

class Flow(nn.Module):
	def __init__(self):
		super(Flow, self).__init__()

	def f(self, x):
		raise NotImplementedError
		return z

	def f_inv(self, z):
		raise NotImplementedError
		return x

	def log_det_jac(self, x):
		raise NotImplementedError
		return 0.0

class IdentityFlow(Flow):
	# useless module! only for debugging.
	def __init__(self):
		super(IdentityFlow, self).__init__()

	def f(self, x):
		z = x
		return z

	def f_inv(self, z):
		x = z
		return x

	def log_det_jac(self, x):
		return 0.0

class CompositeFlow(Flow):
	def __init__(self, flow_list):
		super(CompositeFlow, self).__init__()
		self.subflows = nn.ModuleList(flow_list)

	def f(self, x):
		for i in range(len(self.subflows)):
			x = self.subflows[i].f(x)
		z = x
		return z

	def f_inv(self, z):
		for i in reversed(range(len(self.subflows))):
			z = self.subflows[i].f_inv(z)
		x = z
		return x

	def log_det_jac(self, x):
		res = 0.0
		for i in range(len(self.subflows)):
			res += self.subflows[i].log_det_jac(x) # exploit chain rule
			x = self.subflows[i].f(x)
		return res

class CouplingLayer1D(Flow):
	def __init__(self, full_dim, change_first):
		super(CouplingLayer1D, self).__init__()
		self.change_first = change_first
		self.half_dim = full_dim//2
		hidden_dim = 200
		self.net = nn.Sequential(
			nn.Linear(self.half_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, full_dim)
		)

	def f(self, x):
		'''
		x: [B, full_dim]
		---
		z: [B, full_dim]
		'''
		if self.change_first:
			net_input = x[:, self.half_dim:] # input second half
		else:
			net_input = x[:, :self.half_dim] # input first half
		net_out = self.net(net_input)
		log_scale = net_out[:, :self.half_dim]
		bias = net_out[:, self.half_dim:]
		z = x.clone()
		if self.change_first:
			z[:, :self.half_dim] = torch.exp(log_scale) * z[:, :self.half_dim] + bias
		else:
			z[:, self.half_dim:] = torch.exp(log_scale) * z[:, self.half_dim:] + bias
		return z

	def f_inv(self, z):
		'''
		z: [B, full_dim]
		---
		x: [B, full_dim]
		'''
		if self.change_first:
			net_input = z[:, self.half_dim:] # input second half (unchanged)
		else:
			net_input = z[:, :self.half_dim] # input first half (unchanged)
		net_out = self.net(net_input)
		log_scale = net_out[:, :self.half_dim]
		bias = net_out[:, self.half_dim:]
		x = z.clone()
		if self.change_first:
			x[:, :self.half_dim] = torch.exp(-log_scale) * (x[:, :self.half_dim] - bias)
		else:
			x[:, self.half_dim:] = torch.exp(-log_scale) * (x[:, self.half_dim:] - bias)
		return x

def test1():
	flows = [IdentityFlow() for _ in range(5)]
	cflow = CompositeFlow(flows)
	x = torch.randn((5))
	print(x)
	z = cflow.f(x)
	print(z)
	x_r = cflow.f_inv(z)
	print(x_r)
	print(cflow.log_det_jac(z))

def test2():
	B = 2
	full_dim = 4
	x = torch.randn((B, full_dim))
	flow_1 = CouplingLayer1D(full_dim, True)
	flow_2 = CouplingLayer1D(full_dim, False)
	flow = CompositeFlow([flow_1, flow_2])
	z = flow.f(x)
	x_r = flow.f_inv(z)
	print(x)
	print(z)
	print(x_r)

test2()