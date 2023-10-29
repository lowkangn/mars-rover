'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDiscFactory
from MCTGenerator import MCTGenerator
from time import time
import tracemalloc

class MCTSAgent(object):
	def	__init__(self, env=None, gamma=0.99, c=2, max_iterations=1000):
		self.max_iterations = max_iterations
		self.mct = MCTGenerator(env, c, gamma)

	def step(self, state):
		return

	def solve(self):
		"""
		Insert solving mechanism here.
		"""
		return


def main():
	myEnv = MarsRoverDiscFactory().get_env(level='1', instance='0')
	agent = MCTSAgent(env=myEnv)
	total_reward = 0

	# for step in range(myEnv.horizon):
	# 	action = agent.step(state)
	# 	next_state, reward, done, info = myEnv.step(action)
	# 	total_reward += reward
	# 	print()
	# 	print('step       = {}'.format(step))
	# 	print('state      = {}'.format(state), myEnv.disc_states[state])
	# 	print('action     = {}'.format(action), myEnv.disc_actions[action])
	# 	print('next state = {}'.format(next_state), myEnv.disc_states[next_state])
	# 	print('reward     = {}'.format(reward))
	# 	state = next_state
	# 	if done:
	# 		break
	# print("episode ended with reward {}".format(total_reward))
	# myEnv.close()

if __name__ == "__main__":
	start = time()
	tracemalloc.start()
	main()
	end = time()
	print(f"total time taken: {end - start} (in s)")
	print(f"memory used (current, peak): {tracemalloc.get_traced_memory()} (in bytes)")
	tracemalloc.stop()
