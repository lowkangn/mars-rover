'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDisc
from time import time
import tracemalloc

class Agent(object):
	def	__init__(self, env=None, gamma=0.99, theta=0.00001, max_iterations=10000):
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
		Insert solving mechanism here.
		"""
		return


def main():
	myEnv = MarsRoverDisc(level='1', instance='0')
	agent = Agent(env = myEnv)
	agent.initialize()
	state = myEnv.reset()
	total_reward = 0

	for step in range(myEnv.horizon):
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
