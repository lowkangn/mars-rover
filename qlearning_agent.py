'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDiscFactory
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import tracemalloc
import pandas as pd

class Agent(object):
    def __init__(self, env=None, instance = None, level = None,  decay_rate=0.005, learning_rate=0.8):
        self.env = env
        self.instance = instance
        self.level = level

        # Set of discrete actions for evaluator environment, shape - (|A|)
        self.disc_actions = env.disc_actions
        # Set of discrete states for evaluator environment, shape - (|S|)
        self.disc_states = env.disc_states

        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.qtable = None

    def initialize(self):
        qtable = self.qlearning(total_episodes=1000, max_steps=99, epsilon=0.5)

    def agentsStep(self, state):
        action = np.argmax(self.qtable[state, :])
        return action

    def qlearning(self, total_episodes, max_steps, epsilon,
              max_epsilon = 1.0, min_epsilon = 0.01,  gamma=0.99, plot_every=10):

        # For heatmap
        x_list = []
        y_list = []

        rewards = []   # List of rewards
        tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
        avg_scores = deque(maxlen=total_episodes)   # average scores over every plot_every episodes

        # initialize Q[S, A] arbitrarily
        self.qtable = np.random.uniform(low=-1, high=1, size=(len(self.disc_states), len(self.disc_actions)))

        for episode in range(total_episodes):
            state = self.env.reset()  # Reset the environment to the starting state
            done = False
            total_rewards = 0  # collected reward within an episode
            
            for step in range(max_steps):
                action = self.epsilon_greedy_policy(self.qtable, state, epsilon)
               
                # call the epsilon greedy policy to obtain the actions
                # take the action and observe resulting reward and state
                new_state, reward, done, info = self.env.step(action)
                self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (
                            reward + gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])

                total_rewards += reward
                state = new_state

                # For heatmap
                x_list.append(self.env.disc_states[state][0][0])
                y_list.append(self.env.disc_states[state][0][1])

                if done == True:
                    break
            
            tmp_scores.append(total_rewards)  #for plot
            if (episode % plot_every == 0): #for plot
                avg_scores.append(np.mean(tmp_scores))
            
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-self.decay_rate * episode)  # Reduce epsilon value to encourage exploitation and discourage exploration
            rewards.append(total_rewards)

            if episode % 10 == 0:  # monitor progress
                 print("\rEpisode {}/{}".format(episode, total_episodes), end='')
                 print(" ")
                 print("Total rewards in episode", episode, " is", total_rewards)
                 print(" ")

        # Plot performance in term of convergence
        plt.plot(np.linspace(0,episode,len(avg_scores),endpoint=False), np.asarray(avg_scores))
        plt.xlabel('Episode Number', fontsize=6)
        plt.ylabel('Average Reward (Over %d Episodes)' % plot_every, fontsize=6)
        plt.title("Q-learning average reward during learning(level="+str(self.level)+", instance="+str(self.instance)+")", fontsize=8)

        filename = "Ravg_q-learning_i"+str(self.instance)+"_l"+str(self.level)+".png"
        plt.savefig(filename)
        plt.close()

        # Create heatmap of visited states
        heatmap, xedges, yedges = np.histogram2d(x_list, y_list, bins=(abs(min(x_list)-max(x_list)), abs(min(y_list)-max(y_list))))
        plt.imshow(heatmap, cmap='viridis', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.xlabel("X Position", fontsize=6)
        plt.ylabel("Y Position", fontsize=6)
        plt.title("Q-learning visited position during learning (level="+str(self.level)+", instance="+str(self.instance)+")", fontsize=8)
        plt.colorbar(label="Visit Count")

        filename = "hm_q-learning_i"+str(self.instance)+"_l"+str(self.level)+".png"
        plt.savefig(filename)
        plt.close()

        # print best 100-episode performance 
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

        return self.qtable

    def epsilon_greedy_policy(self, Q, state, epsilon):
        # Q:          : state-action pair
        # State (int) : current state
        # eps (float): epsilon
        action = 0
        if random.uniform(0, 1) > epsilon:  # exploitation
            action = np.argmax(Q[state, :])
        else:  # exploration
            action = random.choice(list(self.disc_actions.keys()))
        return action

def main(instance, level, learning_rate, decay_rate):
    t1 = time.time()
    tracemalloc.start()

    myEnv = MarsRoverDiscFactory().get_env(instance=instance, level=level)
    myEnv.initialize()
    agent = Agent(env=myEnv, instance=instance, level=level,  learning_rate=learning_rate, decay_rate=decay_rate)
    agent.initialize() #qlearning
    state = myEnv.reset()
    total_reward = 0

    for step in range(myEnv.horizon):
        # myEnv.render()
        action = agent.agentsStep(state)
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

    # Keep track of runtime and memory used 
    t2 = time.time()
    runtime = t2-t1
    memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return memory, runtime, total_reward

# Perform hyperparameter grid search
#header = ['Learning rate', 'Decay rate', 'Memory', 'Runtime', 'Total reward'] 
#df = pd.DataFrame([header])
#df.to_excel('Q_HyperparSearch.xlsx', index=False)

#learning_rate = [0.2, 0.4, 0.6, 0.8]
#decay_rate  = [0.001, 0.005, 0.05, 0.1, 0.2]

#for l in learning_rate:
    #for d in decay_rate:

        # Get results
        #memory, runtime, total_reward = main(instance='3c', level='2', learning_rate=l, decay_rate=d)
        #trial_result = [l, d, memory, runtime, total_reward]

        # Update excel sheet
        #df = pd.DataFrame([trial_result])
        #existing_data = pd.read_excel('Q_HyperparSearch.xlsx')
        #updated_data = pd.concat([existing_data, df], ignore_index=False)
        #updated_data.to_excel("Q_HyperparSearch.xlsx", index=False)


# Run all levels/ instances
header = ['Level', 'Instance', 'Memory', 'Runtime', 'Total reward'] 
df = pd.DataFrame([header])
df.to_excel('Qlearning_performance.xlsx', index=False)

levels = ['3']
instances = ['2c', '3c']
learning_rate = 0.8
decay_rate = 0.005

for l in levels:
    for i in instances:

        # Get results
        memory, runtime, total_reward = main(instance=i, level=l, learning_rate=learning_rate, decay_rate=decay_rate)
        trial_result = [l, i, memory, runtime, total_reward]

        # Update excel sheet
        df = pd.DataFrame([trial_result])
        existing_data = pd.read_excel('Qlearning_performance.xlsx')
        updated_data = pd.concat([existing_data, df], ignore_index=False)
        updated_data.to_excel('Qlearning_performance.xlsx', index=False)


# Run single level/ instance
#memory, runtime, total_reward = main(instance='3c', level='2', learning_rate=l, decay_rate=d)
