from typing import Optional, Tuple
from gym import Env
import itertools
import numpy as np
import pickle
from bidict import bidict
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Visualizer.MarsRoverViz import MarsRoverVisualizer
from TransitionModelGenerator import TransitionModelGenerator

"""
Discretized version of Mars Rover enviroment for Level 1.
Adapted from https://github.com/tasbolat1/pyRDDLGym/blob/2b60ec7e6406a335fa4c14496f35a216fa50eb3b/pyRDDLGym/Elevator.py
"""
class MarsRoverDisc(Env):
    def __init__(self, instance='0'): 
        self.base_env = RDDLEnv.RDDLEnv(domain="level 1/domain.rddl", instance=f"level 1/instance{instance}.rddl")
        self.numConcurrentActions = self.base_env.numConcurrentActions
        self.horizon = self.base_env.horizon
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.discount = self.base_env.discount
        
        # set up the environment visualizer
        self.base_env.set_visualizer(MarsRoverVisualizer)

        # global variables
        self.rover_step_size = int(self.base_env.sampler.subs['MAX-STEP'])
        self.x_bound = int(self.base_env.sampler.subs['MAX-X'])
        self.y_bound = int(self.base_env.sampler.subs['MAX-Y'])
        self.mineral_count = int(len(self.base_env.sampler.subs['MINERAL-VALUE']))
        self.mineral_pos = list(zip(self.base_env.sampler.subs['MINERAL-POS-X'], self.base_env.sampler.subs['MINERAL-POS-Y']))
        self.mineral_values = self.base_env.sampler.subs['MINERAL-VALUE']

        self.disc_states = self.init_states()
        self.disc_actions = self.init_actions()

        try:
            f = open(f'level 1/instance{instance}.pickle', 'rb')
            self.Prob = pickle.load(f)
            f.close()
        except:
            model = TransitionModelGenerator(self, instance=instance)
            self.Prob = model.generate_transitions()

        # print(f"The environment extends {self.x_bound} units in the x-direction and {self.y_bound} units in the y-direction from the origin.")

    def init_states(self):
        '''
        Initialise discrete space
        '''
        # each possible position of the rover
        rover_possible_pos = list(itertools.product(np.arange(-self.x_bound, self.x_bound + 1), np.arange(-self.y_bound, self.y_bound + 1)))

        # all combinations of each mineral being harvested/not harvested
        m_harvested_combinations = list(itertools.product(np.arange(0, 2), repeat=self.mineral_count))

        # each state is represented in the form ((rover_pos_x, rover_pos_y), (m1_harvested, m2_harvested, ... ))
        states = list(itertools.product(rover_possible_pos, m_harvested_combinations))
        disc_states = {}

        for i, _v in enumerate(states):
            disc_states[i] = _v
        return bidict(disc_states)
    
    def init_actions(self):
        '''
        Initialise discrete actions
        '''
        disc_actions = {}
        index = 0
        # each action is represented in the form ACTION_NAME|ACTION_VALUE
        for k, _v in self.action_space.items():
            for i in range(_v.n):
                disc_actions[index] = k + '|' + str(_v.start + i)
                index += 1
        return bidict(disc_actions)
    
    def step(self, action):
        cont_action = self.disc2action(action)
        next_state, reward, done, info =  self.base_env.step(cont_action)
        return self.state2disc(next_state), reward, done, info
    
    def reset(self, seed=None):
        state = self.base_env.reset(seed=seed)
        return self.state2disc(state)
    
    def render(self):
        self.base_env.render()
    
    def disc2action(self, a):
        '''
        Converts discrete action into Level 1 Mars Rover environment.
        Input:
            - a (int): action
        Return:
            - a (definition): action that is compatible with Level 1 Mars Rover environment.
        '''
        a_def = self.disc_actions[a]
        a_desc, value = a_def.split('|')
        action = { a_desc: int(value) }
        return action


    def disc2state(self, s):
        '''
        Converts discrete state into Level 1 Mars Rover environment state.
        Input:
            - s (int): action
        Return:
            - s (definition): state that is compatible with Level 1 Mars Rover environment.
        '''
        s_def = self.disc_states[s]
        state = {}
        state['pos-x___d1'] = s_def[0][0]
        state['pos-y___d1'] = s_def[0][1]
        for i in range(0, self.mineral_count):
            state[f'mineral-harvested___m{i + 1}'] = s_def[1][i]

        return state

    def state2disc(self, state):
        rover_pos_x = state['pos-x___d1']
        rover_pos_y = state['pos-y___d1']
        rover_pos = (rover_pos_x, rover_pos_y)
        mineral_harvest = []
        for i in range(1, self.mineral_count + 1):
            mineral_harvest.append(state[f'mineral-harvested___m{i}'])
        disc_state = (rover_pos, tuple(mineral_harvest))

        if disc_state in self.disc_states.inverse:
            return self.disc_states.inverse[disc_state]

        return None
