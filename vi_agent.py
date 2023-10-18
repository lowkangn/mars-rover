import numpy as np
from MarsRoverDisc import MarsRoverDisc

class ValueIterationAgent(object):
    def __init__(self, env=None, gamma=0.99, theta = 0.00001, max_iterations=10000):
        self.env = env
        self.disc_actions = env.disc_actions
        self.disc_states = env.disc_states
        self.Prob = env.Prob

        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

    def initialize(self):
        self.value_policy, self.policy_function = self.solve_value_iteration()

    def step(self, state):
        action = self.policy_function[int(state)]
        return action

    def solve_value_iteration(self):
        '''
        return:
            value_policy (shape - (|S|)): utility value for each state
            policy_function (shape - (|S|), dtype = int64): action policy per state
        '''
        value_policy = np.random.choice(len(self.env.disc_actions), size=len(self.env.disc_states))
        policy_function = np.random.choice(len(self.env.disc_actions), size=len(self.env.disc_states))

        # value_policy = [0] * len(self.disc_states)
    
        # for _ in range(self.max_iterations):
        #     delta = 0
        #     for s in self.disc_states:
        #         v = value_policy[s]
        #         value_policy[s] = max([self.bellman_update(value_policy, s, a, self.gamma)
        #                     for a in self.env.disc_actions])          
        #         delta = max(delta, abs(v - value_policy[s]))
        #     if delta <= self.theta: 
        #         break
        
        # policy_function = [0] * len(self.disc_states)

        # for s in self.disc_states:
        #     policy_function[s] = np.argmax([self.bellman_update(value_policy, s, a, self.gamma)
        #                     for a in self.disc_actions])

        return value_policy, policy_function
    
    def bellman_update(self, V, s, a, discount):
        p, s_, r =  self.Prob[s][a]
        utility = p * (r + (discount * V[s_]))
        return utility


def main():

    env = MarsRoverDisc(instance='0')
    agent = ValueIterationAgent(env=env, gamma=env.discount)
    agent.initialize()
    state = env.reset()
    
    total_reward = 0
    for _ in range(env.horizon):
        env.render()
        action = agent.step(state)
        next_state, reward, terminated, _ = env.step(action)
        
        total_reward += reward
        print()
        print(f'state      = {state}')
        print(f'action     = {action}')
        print(f'next state = {next_state}')
        print(f'reward     = {reward}')
        print(f'total_reward     = {total_reward}')
        state = next_state

    env.close()
    

if __name__ == "__main__":
    main()

