'''
Adapted from agent template given in assignment 2
'''
from MarsRoverDisc import MarsRoverDisc
import numpy as np

myEnv = MarsRoverDisc(instance='0')

class Agent(object):
	def	__init__(self, env=None, gamma=0.99, theta = 0.00001, max_iterations=10000):
		self.env = env
		# Set of discrete actions for evaluator environment, shape - (|A|)
		self.disc_actions = env.disc_actions
		# Set of discrete states for evaluator environment, shape - (|S|)
		self.disc_states = env.disc_states
		# Set of probabilities for transition function for each action from every states, dicitonary of dist[s] = [s', prob, done, info]
		self.Prob = env.Prob

		self.gamma = gamma
		self.theta = theta
		self.max_iterations = max_iterations
		self.value_policy, self.policy_function = None, None

	def initialize(self):
		self.value_policy, self.policy_function = self.solve()


	def step(self, state):
		action = self.policy_function[int(state)]
		return action

	def solve(self):
		"""
		insert solving mechanism here
		"""
		u, u_prime, policy_function = np.zeros(len(self.env.disc_states)), np.zeros(len(self.env.disc_states)), np.zeros(len(self.env.disc_states), dtype=int)
		converged = lambda delta: (delta <= self.theta*(1-self.gamma)/self.gamma)
		delta = 1000.0
		count = 0

		def q_val(s_curr,a, u): 
			temp = self.Prob[s_curr][a]
			p = temp[0]
			s_next = temp[1]
			r = temp[2]
			return p*(r+self.gamma*u[s_next])
	
		while not converged(delta):
			u = u_prime.copy()
			delta = 0
			count+=1
			# print(f"count: {count}")
			for curr_state in range(len(self.env.disc_states)):
				q = [q_val(curr_state, x, u) for x in range(6)]
				q_max = max(q)
				u_prime[curr_state] = q_max
				policy_function[curr_state] = q.index(q_max)
				diff = abs(u_prime[curr_state]-u[curr_state])
				if delta < diff:
					delta = diff

		value_policy = u
		# print(f"end count: {count}")
		# print(value_policy)
		# print(policy_function)
		return value_policy, policy_function


def main():
	agent = Agent(env = myEnv)
	agent.initialize()
	state = myEnv.reset()
	print(state)
	total_reward = 0

	for step in range(myEnv.horizon):
		action = agent.step(state)
		next_state, reward, done, info = myEnv.step(action)
		total_reward += reward
		print()
		print('step       = {}'.format(step))
		print('state      = {}'.format(state))
		print('action     = {}'.format(action))
		print('next state = {}'.format(next_state))
		print('reward     = {}'.format(reward))
		state = next_state
		if done:
			break
	print("episode ended with reward {}".format(total_reward))
	myEnv.close()

main()