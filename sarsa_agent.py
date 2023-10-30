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
		self.q_table = None
		self.policy_function = None

	def initialize(self):
		self.q_table, self.policy_function = self.solve()

	def step(self, state):
		action = self.policy_function[int(state)]
		return action

	def solve(self):
		"""
		Insert solving mechanism here.
		"""
		q_table = np.zeros((len(self.disc_states), len(self.disc_actions)))
		policy_function = np.zeros(len(self.disc_states), dtype=int)
		lr = 0.5
		curr_state = 0
		curr_action = 0

		for _ in range(self.max_iterations):
			delta = 0
			p, next_state, r = self.Prob[curr_state][curr_action]
			next_action = policy_function[next_state]
			updated_q =  q_table[curr_state][curr_action] + lr * (r + self.gamma*q_table[next_state][next_action] - q_table[curr_state][curr_action])
			delta = abs(q_table[curr_state][curr_action] - updated_q)
			q_table[curr_state][curr_action] = updated_q
			curr_state = next_state
			curr_action = next_action
			
			# if delta <= self.theta: # termination
			# 	break
		return q_table, policy_function
	

def main():
	myEnv = MarsRoverDiscFactory().get_env(level='1', instance='0')
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


if __name__ == "__main__":
	start = time()
	tracemalloc.start()
	main()
	end = time()
	print(f"total time taken: {end - start} (in s)")
	print(f"memory used (current, peak): {tracemalloc.get_traced_memory()} (in bytes)")
	tracemalloc.stop()
