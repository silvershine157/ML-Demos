# Purpose of this script is to clarify the relations and underlying assumptions of RL concepts

import numpy as np

class MarkovRewardProcess(object):
	def __init__(self, Ns, P, R, gamma):
		'''
		Ns: int
		P: [Ns, Ns]
		R: [Ns]
		gamma: float 0~1
		'''
		self.Ns = Ns
		self.P = P
		self.R = R
		self.gamma = gamma
		self.state = None

	def initialize(self, start=0):
		self.state = start
		return self.state

	def step(self):
		next_state = np.random.choice(np.arange(self.Ns), p=self.P[self.state, :])
		self.state = next_state
		reward = self.R[self.state]
		return next_state, reward

	def create_random(Ns):
		P = np.random.dirichlet(10.0*np.ones(Ns), Ns)
		R = np.random.rand(Ns)
		gamma = 0.5
		return MarkovRewardProcess(Ns, P, R, gamma)

class evaluate:
	class mrp:
		def mc(mrp):
			# first visit monte carlo
			episodes = 100
			horizon = 30 # as an approximation to infinity
			visit_sum = np.zeros(mrp.Ns, np.int)
			return_sum = np.zeros(mrp.Ns)
			for ep in range(episodes):
				discount = np.zeros(mrp.Ns)
				visited = np.zeros(mrp.Ns, dtype=np.bool)
				mrp.initialize()
				for t in range(horizon):
					state, reward = mrp.step()
					if not visited[state]:
						visit_sum[state] += 1
						visited[state] = True
						discount[state] = 1.0
					return_sum[state] += discount[state]*reward
					discount = mrp.gamma*discount
			V = return_sum/visit_sum
			return V

		def analytic(mrp):
			pass
		def iterative(mrp):
			pass


def test1():
	# MRP simulation test
	Ns = 3
	mrp = MarkovRewardProcess.create_random(Ns)
	mrp.initialize()
	for i in range(5):
		state, reward = mrp.step()
		print(state, reward)

def test2():
	# compare MRP evaluation methods
	Ns = 3
	mrp = MarkovRewardProcess.create_random(Ns)
	V_mc = evaluate.mrp.mc(mrp)
	print(V_mc)

test2()

