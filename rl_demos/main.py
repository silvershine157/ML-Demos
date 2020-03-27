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

def test1():
	# MRP evaluation
	Ns = 3
	mrp = MarkovRewardProcess.create_random(Ns)
	mrp.initialize()
	for i in range(5):
		state, reward = mrp.step()
		print(state, reward)

test1()