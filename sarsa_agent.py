'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDiscFactory
import numpy as np
from time import time
import time
import tracemalloc
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

class Agent(object):
	def	__init__(self, env=None, instance = None, level = None,  decay_rate=0.005, learning_rate=0.8, gamma=0.99, theta=0.00001, max_iterations=500):
		self.env = env
		self.instance = instance
		self.level = level
	
		# Set of discrete actions for evaluator environment, shape - (|A|)
		self.disc_actions = env.disc_actions
		# Set of discrete states for evaluator environment, shape - (|S|)
		self.disc_states = env.disc_states
		# Set of probabilities for transition function for each action from every states, dictionary of dist[s] = [s', prob, done, info]
		self.Prob = env.Prob

		self.gamma = gamma
		self.theta = theta
		self.max_iterations = max_iterations
		self.decay_rate = decay_rate
		self.learning_rate = learning_rate
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
		x_list = []
		y_list = []
		avg_scores = deque(maxlen=self.max_iterations)
		plot_every = 10
		
		def greedy(s, e):
			a = 0
			if np.random.uniform(0, 1) < e:
				action = np.random.randint(0,len(self.disc_actions))
			else:
				action = np.argmax(q_table[s])
			return action

		q_table = np.random.rand(len(self.disc_states), len(self.disc_actions))
		lr = self.learning_rate
		decay_rate = self.decay_rate
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
				
				x_list.append(self.env.disc_states[curr_state][0][0])
				y_list.append(self.env.disc_states[curr_state][0][1])
				
			self.ep_rewards.append(ep_reward)
		
			if (iter % plot_every == 0): #for plot
				avg_scores.append(np.mean(self.ep_rewards))

			print(f"ep{iter} ended with reward {ep_reward}")
			epsilon = 0.01 + (1 - 0.01) * np.exp(-decay_rate * iter)
		print("end learning")
		
        # plot performance
		plt.plot(np.linspace(0,iter,len(avg_scores),endpoint=False), np.asarray(avg_scores))
		plt.xlabel('Episode Number', fontsize=6)
		plt.ylabel('Average Reward (Over %d Episodes)' % plot_every, fontsize=6)
		plt.title("SARSA average reward during learning (level="+str(self.level)+", instance="+str(self.instance)+")", fontsize= 8)
		filename = "SARSA_i"+str(self.instance)+"_l"+str(self.level)+".png"
		plt.savefig(filename)
		
        # Create heatmap of visited states
		heatmap, xedges, yedges = np.histogram2d(x_list, y_list, bins=(abs(min(x_list)-max(x_list)), abs(min(y_list)-max(y_list))))
		plt.imshow(heatmap, cmap='viridis', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
		plt.xlabel("X Position", fontsize=6)
		plt.ylabel("Y Position", fontsize=6)
		plt.title("SARSA visited position during learning (level="+str(self.level)+", instance="+str(self.instance)+")", fontsize= 8)
		plt.colorbar(label="Visit Count")
		
		filename = "hm_SARSA_i"+str(self.instance)+"_l"+str(self.level)+".png"
		plt.savefig(filename)
		plt.close()
		
        # print best 100-episode performance
		print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

		return q_table
	

def main(instance, level, learning_rate, decay_rate):
	t1 = time.time()
	tracemalloc.start()
	
	myEnv = MarsRoverDiscFactory().get_env(level=level, instance=instance)
	myEnv.initialize()
	agent = Agent(env = myEnv, level=level, instance=instance, learning_rate=learning_rate, decay_rate=decay_rate)
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
			myEnv.render()
			break
	print("episode ended with reward {}".format(total_reward))
	
	t2 = time.time()
	runtime = t2-t1
	memory = tracemalloc.get_traced_memory()
	tracemalloc.stop()

	myEnv.save_render()
	myEnv.close()
	
	return memory, runtime, total_reward

# Perform hyperparameter grid search
#header = ['Learning rate', 'Decay rate', 'Memory', 'Runtime', 'Total reward'] 
#df = pd.DataFrame([header])
#df.to_excel('SARSA_HyperparSearch.xlsx', index=False)

#learning_rate = [0.4, 0.6, 0.8]
#decay_rate  = [0.001, 0.005, 0.05, 0.1]

#for l in learning_rate:
    #for d in decay_rate:

        # Get results
        #memory, runtime, total_reward = main(instance='3c', level='2', learning_rate=l, decay_rate=d)
        #trial_result = [l, d, memory, runtime, total_reward]

        # Update excel sheet
        #df = pd.DataFrame([trial_result])
        #existing_data = pd.read_excel('SARSA_HyperparSearch.xlsx')
        #updated_data = pd.concat([existing_data, df], ignore_index=False)
        #updated_data.to_excel("SARSA_HyperparSearch.xlsx", index=False)


# Run all levels/ instances
header = ['Level', 'Instance', 'Memory', 'Runtime', 'Total reward'] 
df = pd.DataFrame([header])
df.to_excel('SARSA_performance.xlsx', index=False)

# levels = ['1', '2', '3']
levels = ['1', '2']
instances = ['0', '1c', '2c', '3c']
learning_rate = 0.2
decay_rate = 0.001

# for l in levels:
#     for i in instances:

#         # Get results
#         memory, runtime, total_reward = main(instance=i, level=l, learning_rate=learning_rate, decay_rate=decay_rate)
#         trial_result = [l, i, memory, runtime, total_reward]

#         # Update excel sheet
#         df = pd.DataFrame([trial_result])
#         existing_data = pd.read_excel('SARSA_performance.xlsx')
#         updated_data = pd.concat([existing_data, df], ignore_index=False)
#         updated_data.to_excel('SARSA_performance.xlsx', index=False)


# Run single level/ instance
memory, runtime, total_reward = main(instance='3c', level='2', learning_rate=learning_rate, decay_rate=decay_rate)