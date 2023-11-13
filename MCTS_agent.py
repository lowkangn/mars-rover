'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDiscFactory
from MCTGenerator import MCTGenerator
from time import time
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt

class MCTSAgent(object):
	def	__init__(self, env=None, gamma=0.99, max_iterations=100, max_depth=25):
		self.level, self.instance = env.level, env.instance
		self.max_iterations = max_iterations
		self.mct = MCTGenerator(env, gamma, max_depth)
		self.next_step = 0
		self.horizon = env.horizon

	def step(self):
		action = self.mct.steps[self.next_step]
		self.next_step += 1
		return action

	def solve(self):
		"""
		Insert solving mechanism here.
		"""
		for _ in range(self.horizon):
			for _ in range(self.max_iterations):
				selected = self.mct.select()

				selected, reward = self.mct.expand(selected)

				r = self.mct.simulate(reward)
				self.mct.update(selected, r)

			print(self.mct.next_action())

		# Create heatmap of visited states
		heatmap, xedges, yedges = np.histogram2d(self.mct.x_list, self.mct.y_list, bins=(abs(min(self.mct.x_list)-max(self.mct.x_list)), abs(min(self.mct.y_list)-max(self.mct.y_list))))
		plt.imshow(heatmap, cmap='viridis', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
		plt.xlabel("X Position", fontsize=6)
		plt.ylabel("Y Position", fontsize=6)
		plt.title("MCTS visited position during rollouts (level="+self.level+", instance="+self.instance+")", fontsize= 8)
		plt.colorbar(label="Visit Count")
			
		filename = "hm_MCTS_i"+str(self.instance)+"_l"+str(self.level)+".png"
		plt.savefig(filename)
		plt.close()


def main():
	myEnv = MarsRoverDiscFactory().get_env(level='3', instance='4c')
	agent = MCTSAgent(env=myEnv)
	agent.solve()
	state = myEnv.reset()
	total_reward = 0

	for step in range(myEnv.horizon):
		myEnv.render()
		action = agent.step()
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
	myEnv.save_render()
	myEnv.close()

if __name__ == "__main__":
	start = time()
	tracemalloc.start()
	main()
	end = time()
	print(f"total time taken: {end - start} (in s)")
	print(f"memory used (current, peak): {tracemalloc.get_traced_memory()} (in bytes)")
	tracemalloc.stop()
