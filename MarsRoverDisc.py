from gym import Env
import itertools
import numpy as np
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Visualizer.MarsRoverViz import MarsRoverVisualizer

"""
Discretized version of Mars Rover enviroment for Level 1.
"""
class MarsRoverDisc(Env):
    def __init__(self, instance='0'): 
        self.base_env = RDDLEnv.RDDLEnv(domain="level 1/domain.rddl", instance=f"level 1/instance{instance}.rddl")
        self.numConcurrentActions = self.base_env.numConcurrentActions
        self.horizon = self.base_env.horizon
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
        # set up the environment visualizer
        self.base_env.set_visualizer(MarsRoverVisualizer)

        # global variables
        self.rover_step_size = int(self.base_env.sampler.subs['MAX-STEP'])
        self.x_bound = int(self.base_env.sampler.subs['MAX-X'])
        self.y_bound = int(self.base_env.sampler.subs['MAX-Y'])
        self.mineral_count = int(len(self.base_env.sampler.subs['MINERAL-VALUE']))

        # self.disc_states = self.init_states()
        # self.disc_actions = self.init_actions()

        # print(f"The environment extends {self.x_bound} units in the x-direction and {self.y_bound} units in the y-direction from the origin.")

    def init_states(self):
        '''
        Initialise discrete space
        '''
        # each possible position of the rover
