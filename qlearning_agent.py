'''
Adapted from agent template given in assignment 2.
'''
from MarsRoverDisc import MarsRoverDiscFactory
import numpy as np
import random


class Agent(object):
    def __init__(self, env=None, gamma=0.99, theta=0.0001, max_iterations=10000):
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
        self.qtable = None

    def initialize(self):
        qtable = self.qlearning(total_episodes=2000, max_steps=99, epsilon=0.5, learning_rate=0.8)
        for i in range(len(qtable)):
            print(qtable[i])


    def agentsStep(self, state):
        action = np.argmax(self.qtable[state, :])
        return action

    def qlearning(self, total_episodes, max_steps, epsilon, learning_rate,
              max_epsilon = 1.0, min_epsilon = 0.01, decay_rate = 0.005,  gamma=0.99):

        # initialize Q[S, A] arbitrarily
        #self.qtable = np.zeros((len(self.disc_states), len(self.disc_actions)))
        self.qtable = np.random.uniform(low=-1, high=1, size=(len(self.disc_states), len(self.disc_actions)))

        for episode in range(total_episodes):
            state = self.env.reset()  # Reset the environment to the starting state
            done = False
            total_rewards = 0  # collected reward within an episode
            if episode % 10 == 0:  # monitor progress
                 print("\rEpisode {}/{}".format(episode, total_episodes), end='')

            for step in range(max_steps):
                action = self.epsilon_greedy_policy(self.qtable, state, epsilon)
                # call the epsilon greedy policy to obtain the actions
                # take the action and observe resulting reward and state
                new_state, reward, done, info = self.env.step(action)

                #print()
                #print('step       = {}'.format(step))
                #print('state      = {}'.format(state), self.env.disc_states[state])
                #print('action     = {}'.format(action), self.env.disc_actions[action])
                #print('next state = {}'.format(new_state), self.env.disc_states[new_state])
                #print('reward     = {}'.format(reward))

                # print('self.qtable[state, action] before: ' + str(self.qtable[state, action]))
                self.qtable[state, action] = self.qtable[state, action] + learning_rate * (
                            reward + gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])
                # print('self.qtable[state, action] after: ' + str(self.qtable[state, action]))
                # print(self.qtable[state, action])
                # print(total_rewards)
                # print('total_rewards afterwards')
                total_rewards += reward
                # print(total_rewards)
                # print('--')
                state = new_state
                if done == True:
                    break
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)  # Reduce epsilon value to encourage exploitation and discourage exploration
            print("Total rewards in episode", episode, " is", total_rewards)
            # print(self.qtable)

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

def main():
    level = '2'
    instance = '0'
    myEnv = MarsRoverDiscFactory().get_env(level, instance)
    myEnv.initialize()
    agent = Agent(env=myEnv)
    agent.initialize() #qlearning
    state = myEnv.reset()
    total_reward = 0

    for step in range(myEnv.horizon):
        myEnv.render()
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


main()