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

class MarkovDecisionProcess(object):
	def __init__(self, Ns, Na, P, R, gamma):
		'''
		Ns: int
		Na: int
		P: [Ns, Na, Ns]
		R: [Ns, Na]
		gamma: float 0~1
		'''
		self.Ns = Ns
		self.Na = Na
		self.P = P
		self.R = R
		self.gamma = gamma

	def initialize(self, start=0):
		self.state = start
		return self.state

	def step(self, action):
		reward = self.R[self.state, action]
		next_state = np.random.choice(np.arange(self.Ns), p=self.P[self.state, action, :])
		return next_state, reward

	def create_random(Ns, Na):
		P = np.random.dirichlet(10.0*np.ones(Ns), (Ns, Na))
		R = np.random.rand(Ns, Na)
		gamma = 0.5
		return MarkovDecisionProcess(Ns, Na, P, R, gamma)


class evaluate:
	class mrp:
		def mc(mrp):
			# first visit monte carlo
			episodes = 1000
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
					return_sum += reward*discount
					discount = mrp.gamma*discount
			V = return_sum/visit_sum
			return V

		def analytic(mrp):
			# analytically using matrix inverse
			temp = np.linalg.inv(np.eye(mrp.Ns) - mrp.gamma*np.transpose(mrp.P))
			V = np.matmul(temp, mrp.R)
			return V

		def iterative(mrp):
			V = np.zeros(mrp.Ns)
			for _ in range(20):
				V = mrp_bellman(V, mrp.P, mrp.R, mrp.gamma)
				#print(V)
			return V

def mrp_bellman(V, P, R, gamma):
	return R + gamma*np.matmul(np.transpose(P), V)

def test1():
	# MRP simulation
	Ns = 3
	mrp = MarkovRewardProcess.create_random(Ns)
	mrp.initialize()
	for i in range(5):
		state, reward = mrp.step()
		print(state, reward)

def test2():
	# compare MRP evaluation methods
	Ns = 5
	mrp = MarkovRewardProcess.create_random(Ns)
	print("< MRP evaluation test >")
	print('MC')
	V_mc = evaluate.mrp.mc(mrp)
	print(V_mc)
	print('analytic')
	V_analytic = evaluate.mrp.analytic(mrp)
	print(V_analytic)
	print('iterative')
	V_iter = evaluate.mrp.iterative(mrp)
	print(V_iter)

def random_policy(Ns, Na):
	'''
	policy: [Ns, Na]
	'''
	policy = np.random.dirichlet(10.0*np.ones(Na), Ns)
	return policy

def test3():
	# MDP simulation
	Ns = 3
	Na = 5
	mdp = MarkovDecisionProcess.create_random(Ns, Na)
	policy = random_policy(Ns, Na)
	state = mdp.initialize()
	for _ in range(5):
		action = np.random.choice(np.arange(Na), p=policy[state, :])
		print(state, action)
		state, reward = mdp.step(action)
		print('reward: {:g}'.format(reward))

test3()

