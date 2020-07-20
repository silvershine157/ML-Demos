import torch

'''
<convention>
f: x -> z
f_inv: z -> x
'''

class Flow(nn.Module):
	def __init__(self):
		super(IdentityFlow, self).__init__()

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

	def f(self, z):
		x = z
		return x

	def f_inv(self, x):
		z = x
		return z

	def log_det_jac(self, z):
		return 0.0

class CompositeFlow(Flow):
	def __init__(self, flow_list):
		super(CompositeFlow, self).__init__()
		self.subflows = nn.ModuleList(flow_list)

	def f(self, z):
		x = z
		for i in range(len(self.subflows)):
			x = self.subflows[i].f(x)
		return x

	def f_inv(self, x):
		z = x
		for i in reversed(range(len(self.subflows))):
			z = self.subflows[i].f_inv(z)
		return z

	def log_det_jac_wrt_z(self, z):
		res = 0.0
		for i in range(len(self.subflows)):
			res += self.subflows[i].log_det_jac(z) # exploit chain rule
			z = self.subflows[i].f(z)
		return res