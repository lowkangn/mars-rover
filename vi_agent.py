'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDiscFactory
import numpy as np
from time import time
import tracemalloc

class Agent(object):
	def	__init__(self, env=None, gamma=0.99, theta=0.00001, max_iterations=10000):
		self.env = env
		# Set of discrete actions for evaluator environment, shape - (|A|)
		self.disc_actions = env.disc_actions
		# Set of discrete states for evaluator environment, shape - (|S|)
		self.disc_states = env.disc_states
		# Set of probabilities for transition function for each action from every states, dictionary of dist[s] = [s', prob, done, info]
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
		Insert solving mechanism here.
		"""
		value_policy, policy_function = np.zeros(len(self.env.disc_states)), np.zeros(len(self.env.disc_states), dtype=int)
	
		for _ in range(self.max_iterations):
			delta = 0
			for curr_state in range(len(self.env.disc_states)):
				u = value_policy[curr_state]
				q = [self.bellman_update(curr_state, a, value_policy) for a in self.env.disc_actions]
				q_max = max(q)
				delta = max(delta, abs(u - q_max))

				value_policy[curr_state] = q_max
				policy_function[curr_state] = q.index(q_max)		

			if delta <= self.theta: # termination
				break

		return value_policy, policy_function
	
	def bellman_update(self, s_curr, a, u):
			p, s_next, r = self.Prob[s_curr][a]
			return p * (r + self.gamma * u[s_next])

def main():
	myEnv = MarsRoverDiscFactory().get_env(level='3', instance='0')
	myEnv.initialize()
	agent = Agent(env = myEnv)
	agent.initialize()
	state = myEnv.reset()
	total_reward = 0

	for step in range(myEnv.horizon):
		myEnv.render()
		action = agent.step(state)
		next_state, reward, done, info = myEnv.step(action)
		total_reward += reward
		print()
		print('step       = {}'.format(step))
		print('state      = {}'.format(state), myEnv.disc_states[state])
		print('action     = {}'.format(action), myEnv.disc_actions[action])
		print('next state = {}'.format(next_state), myEnv.disc_states[next_state])
		print('reward     = {}'.format(reward))
		state = next_state
		if done:
			break
	print("episode ended with reward {}".format(total_reward))
	myEnv.close()

start = time()
tracemalloc.start()
main()
end = time()
print(end - start)
print(tracemalloc.get_traced_memory())
tracemalloc.stop()
