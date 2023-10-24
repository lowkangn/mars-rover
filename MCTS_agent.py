'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDisc

class Agent(object):
	def	__init__(self, env=None, gamma=0.99, max_iterations=1000):
		self.env = env

		self.gamma = gamma
		self.max_iterations = max_iterations
		self.policy_function = None

	def initialize(self):
		self.policy_function = self.solve()

	def step(self, state):
		return

	def solve(self):
		"""
		Insert solving mechanism here.
		"""
		return


def main():
	myEnv = MarsRoverDisc(instance='0')
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

main()