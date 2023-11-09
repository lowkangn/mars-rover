'''
Adapted from agent template given in assignment 2
'''
from MarsRoverDisc import MarsRoverDiscFactory
import random
import numpy as np
import time
import matplotlib.pyplot as plt

class Agent(object):
	def	__init__(self, env=None, gamma=0.99, theta = 0.00001, max_iterations=10000, plot = False):
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
		self.plot = plot
		self.delta_values =[]

	def initialize(self):
		self.value_policy, self.policy_function = self.solve()


	def step(self, state):
		action = self.policy_function[int(state)]
		return action

	def solve(self):
		"""
		insert solving mechanism here
		"""
            
		num_states = len(self.env.disc_states)
		num_actions = len(self.env.disc_actions)
		
		# Initialize a random policy
		policy_function = np.random.randint(0, num_actions, size=num_states)
		
		for _ in range(self.max_iterations):
			# Policy Evaluation
			value_policy = self.policy_evaluation(policy_function)
			
			policy_stable = True
			for curr_state in range(num_states):
				# Policy Improvement
				old_action = policy_function[curr_state]
				new_action = self.policy_improvement(curr_state, value_policy)
				policy_function[curr_state] = new_action
				
				if old_action != new_action:
					policy_stable = False
			
			if policy_stable:
				break
		
		if self.plot == True:
			iteration_x = range(1, len(self.delta_values) + 1)

			plt.plot(iteration_x, self.delta_values, label='Delta values') #Plot delta values and line for theta
			plt.axhline(y=self.theta, color='red', linestyle='--', label='Theta')

			plt.xlabel('Iterations')
			plt.ylabel('Delta')
			plt.title('Convergence of delta values')
			plt.legend()
			plt.savefig(f'L{self.env.level} I{self.env.instance} delta convergence.png')

		return value_policy, policy_function
	
	def policy_evaluation(self, policy):
		
		value_policy = np.zeros(len(self.env.disc_states))
		
		while True:
			delta = 0
			for curr_state in range(len(self.env.disc_states)):
				u = value_policy[curr_state]
				a = policy[curr_state]
				q = self.bellman_update(curr_state, a, value_policy)
				delta = max(delta, abs(u - q))
				value_policy[curr_state] = q
			
			self.delta_values.append(delta)
			
			if delta <= self.theta:  # Termination
				break
		
		return value_policy

	def policy_improvement(self, s_curr, value_policy):
		# Find the action that maximizes the expected return
		q_values = [self.bellman_update(s_curr, a, value_policy) for a in self.env.disc_actions]
		
		return np.argmax(q_values)

	def bellman_update(self, s_curr, a, value_policy):
		p, s_next, r = self.Prob[s_curr][a]
		
		return p * (r + self.gamma * value_policy[s_next])

def main():
	t1 = time.time()
	level = '1'
	instance = '0'

	myEnv = MarsRoverDiscFactory().get_env(level, instance)
	myEnv.initialize()
	agent = Agent(env = myEnv, plot=True)
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
	
	t2 = time.time()
	runtime = t2-t1

	print()
	print("summary of level {}, instance {}".format(level, instance))
	print("episode ended with reward {}".format(total_reward))
	print("total runtime is {} seconds".format(runtime))

	myEnv.close()

main()