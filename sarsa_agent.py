'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDiscFactory
import numpy as np
from time import time
import tracemalloc
from matplotlib import pyplot as plt

class Agent(object):
	def	__init__(self, env=None, gamma=0.99, theta=0.00001, max_iterations=5000):
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
		self.ep_rewards = []

	def initialize(self):
		self.q_table = self.solve()

	def step(self, state):
		action = np.argmax(self.q_table[int(state)])
		return action

	def solve(self):
		"""
		Insert solving mechanism here.
		"""
		def greedy(s, e):
			a = 0
			if np.random.uniform(0, 1) < e:
				action = np.random.randint(0,len(self.disc_actions))
			else:
				action = np.argmax(q_table[s])
			return action

		q_table = np.random.rand(len(self.disc_states), len(self.disc_actions))
		lr = 0.5
		decay_rate = 0.005
		epsilon = 0.5

		for iter in range(self.max_iterations):
			if epsilon < 0.01 or epsilon > 1: 
				break
			done = False
			ep_reward = 0
			curr_state = self.env.reset()
			curr_action = greedy(curr_state, epsilon)
			#print(f"ep{iter} start")
			while not done:
				next_state, r, done, info = self.env.step(curr_action)
				ep_reward += r
				next_action = greedy(next_state, epsilon)
				q_table[curr_state][curr_action] = q_table[curr_state][curr_action] + lr * (r + self.gamma*q_table[next_state][next_action] - q_table[curr_state][curr_action])
				curr_state = next_state
				curr_action = next_action
			self.ep_rewards.append(ep_reward)
			print(f"ep{iter} ended with reward {ep_reward}")
			epsilon = 0.01 + (1 - 0.01) * np.exp(-decay_rate * iter)
		print("end learning")
		# print(q_table)
		return q_table
	

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
	plt.plot(range(0,len(agent.ep_rewards)), agent.ep_rewards)
	plt.savefig(f'L{agent.env.level} I{agent.env.instance} ep rewards.png')
	myEnv.close()


if __name__ == "__main__":
	start = time()
	tracemalloc.start()
	main()
	end = time()
	print(f"total time taken: {end - start} (in s)")
	print(f"memory used (current, peak): {tracemalloc.get_traced_memory()} (in bytes)")
	tracemalloc.stop()
