import numpy as np
from rl_utils import *

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


def test4():
	# MDP + stationary policy -> MRP
	Ns = 3
	Na = 5
	mdp = MarkovDecisionProcess.create_random(Ns, Na)
	policy = random_policy(Ns, Na)
	mrp = mdp_policy_to_mrp(mdp, policy)
	

test4()

