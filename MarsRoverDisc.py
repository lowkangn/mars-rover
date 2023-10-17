from gym import Env
import itertools
import numpy as np
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Visualizer.MarsRoverViz import MarsRoverVisualizer

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

        self.disc_states = self.init_states()
        self.disc_actions = self.init_actions()

        # print(f"The environment extends {self.x_bound} units in the x-direction and {self.y_bound} units in the y-direction from the origin.")

        print(self.disc_states)

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
        return disc_states
    
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
        return disc_actions
