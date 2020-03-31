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
		self.state = next_state
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

	class mdp_policy:
		def mc(mdp, policy):
			# first visit monte carlo
			episodes = 1000
			horizon = 30
			visit_sum = np.zeros(mdp.Ns, np.int)
			return_sum = np.zeros(mdp.Ns)
			for ep in range(episodes):
				discount = np.zeros(mdp.Ns)
				visited = np.zeros(mdp.Ns, dtype=np.bool)
				state = mdp.initialize()
				for t in range(horizon):
					if not visited[state]:
						visited[state] = True
						visit_sum[state] += 1
						discount[state] = 1.0
					action = sample_action(policy, state)
					state, reward = mdp.step(action)
					return_sum += reward*discount
					discount = mdp.gamma*discount
			V = return_sum/visit_sum
			return V

def mdp_policy_to_mrp(mdp, policy):
	'''
	policy: [Ns, Na]
	P: [Ns, Na, Ns]
	R: [Ns, Na]
	---
	P_pi: [Ns, Na]
	R_pi: [Ns]
	'''
	P_pi = np.einsum('sa,san->sn', policy, mdp.P)
	R_pi = np.einsum('sa,sa->s', policy, mdp.R)
	mrp = MarkovRewardProcess(mdp.Ns, P_pi, R_pi, mdp.gamma)
	return mrp

def mrp_bellman(V, P, R, gamma):
	return R + gamma*np.matmul(np.transpose(P), V)

def sample_action(policy, state):
	'''
	policy: [Ns, Na]
	'''
	Na = policy.shape[1]
	action = np.random.choice(np.arange(Na), p=policy[state, :])
	return action

def random_policy(Ns, Na):
	'''
	policy: [Ns, Na]
	'''
	policy = np.random.dirichlet(1.0*np.ones(Na), Ns)
	return policy
