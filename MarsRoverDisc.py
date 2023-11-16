from gym import Env
import itertools
import numpy as np
import pickle
import glob
import os
import re
from bidict import bidict
from pyRDDLGym import RDDLEnv
from PIL import Image
from pyRDDLGym.Visualizer.MarsRoverViz import MarsRoverVisualizer
from pathlib import Path
from TransitionModelGenerator import TransitionModelGenerator

"""
Discretized version of Mars Rover enviroment.
Adapted from https://github.com/tasbolat1/pyRDDLGym/blob/2b60ec7e6406a335fa4c14496f35a216fa50eb3b/pyRDDLGym/Elevator.py
"""

class MarsRoverDiscFactory():
    """
    factory
    """
    def __init__(self):
        self.lvldict = { '1': LevelOneEnv, '2': LevelTwoEnv, '3':LevelThreeEnv }

    def get_env(self, level, instance): 
        if level not in self.lvldict:
            raise Exception("no such level")
        return self.lvldict[level](instance)

class MarsRoverDisc(Env):
    def __init__(self, level='1', instance='0'):
        self.level, self.instance = level, instance
        self.base_env = RDDLEnv.RDDLEnv(domain=f"level {level}/domain.rddl", instance=f"level {level}/instance{instance}.rddl")
        self.numConcurrentActions = self.base_env.numConcurrentActions
        self.horizon = self.base_env.horizon
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.discount = self.base_env.discount
        
        # set up the environment visualizer
        self._n_frame = 0
        self.render_path = f'output/level {level}/instance{instance}'
        self.save_path = os.path.join(self.render_path, 'marsrover' + '_{}' + '.png')
        self.base_env.set_visualizer(MarsRoverVisualizer)
        Path(self.render_path).mkdir(parents=True, exist_ok=True)

        # global variables (applicable to all levels)
        self.mineral_count = int(len(self.base_env.sampler.subs['MINERAL-VALUE']))
        self.mineral_pos = list(zip(self.base_env.sampler.subs['MINERAL-POS-X'], self.base_env.sampler.subs['MINERAL-POS-Y']))
        self.mineral_values = self.base_env.sampler.subs['MINERAL-VALUE']
        self.mineral_areas = self.base_env.sampler.subs['MINERAL-AREA']

        self.disc_states = bidict({})
        self.disc_actions = self.init_actions()
        self.Prob = {}

    def initialize(self):
        # do not use this for larger instances
        self.disc_states = self.init_states()
        self.Prob = self.generate_tm()

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
    
    def generate_tm(self):
        try:
            with open(f'level {self.level}/instance{self.instance}.pickle', 'rb') as f:
                tm =  pickle.load(f)
                f.close()
                return tm
        except:
            model = TransitionModelGenerator().get_tm(self)
            return model.generate_transitions()
    
    def step(self, action):
        cont_action = self.disc2action(action)
        next_state, reward, done, info =  self.base_env.step(cont_action)
        return self.state2disc(next_state), reward, done, info
    
    def reset(self, seed=None):
        state = self.base_env.reset(seed=seed)
        return self.state2disc(state)
    
    def render(self):
        file_path = self.save_path.format(str(self._n_frame).rjust(10, '0'))
        image = self.base_env.render(to_display=False)
        image.save(file_path)
        self._n_frame += 1 

    def save_render(self):       
        load_path = self.save_path.format('*')

        def getOrder(frame):
            return int(re.search('\d+', frame).group(0))

        files = glob.glob(load_path)
        files.sort(key=getOrder)

        # images = map(Image.open, glob.glob(load_path))
        images = map(Image.open, files)
        
        save_path = os.path.join(self.render_path, 'marsrover.gif')
        frame0 = next(images, None)
        if frame0 is not None:
            frame0.save(fp=save_path,
                        format='GIF',
                        append_images=images,
                        save_all=True,
                        duration=100,
                        loop=False) 
    
    def disc2action(self, a):
        '''
        Converts discrete action into the Mars Rover environment.
        Input:
            - a (int): action
        Return:
            - a (definition): action that is compatible with the Mars Rover environment.
        '''
        a_def = self.disc_actions[a]
        a_desc, value = a_def.split('|')
        action = { a_desc: int(value) }
        return action

    def disc2state(self, s):
        '''
        Converts discrete state into the Mars Rover environment state.
        Input:
            - s (int): state
        Return:
            - s (definition): state that is compatible with the Mars Rover environment.
        '''
        pass

    def state2disc(self, s):
        '''
        Converts the Mars Rover environment state into discrete state.
        Input:
            - s (defintion): state that is compatible with the Mars Rover environment.
        Return:
            - s (definition): state
        '''
        pass

"""
For levels 1 and 2, where the rover's movement is based on displacement.
"""  
class DispMarsRoverDisc(MarsRoverDisc):
    def __init__(self, level='1', instance='0'): 
        super().__init__(level, instance)
        # level specific global variables
        self.x_bound = int(self.base_env.sampler.subs['MAX-X'])
        self.y_bound = int(self.base_env.sampler.subs['MAX-Y'])
        self.rover_step_size = int(self.base_env.sampler.subs['MAX-STEP'][0])

    def disc2state(self, s):
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
        disc_state = rover_pos, tuple(mineral_harvest)

        if disc_state in self.disc_states.inverse:
            return self.disc_states.inverse[disc_state]
        else:
            index = len(self.disc_states)
            self.disc_states[index] = disc_state
            return index
    
class LevelOneEnv(DispMarsRoverDisc):
    def __init__(self, instance='0'): 
        super().__init__('1', instance)

class LevelTwoEnv(DispMarsRoverDisc):
    def __init__(self, instance='0'): 
        super().__init__('2', instance)

class LevelThreeEnv(MarsRoverDisc):
    def __init__(self, instance='0'):
        super().__init__('3', instance)
        # level specific global variables
        self.rover_max_power = int(self.base_env.sampler.subs['MAX-POWER'])
        self.rover_max_vel = int(self.base_env.sampler.subs['MAX-VEL'])
        self.x_bound = self.horizon * self.rover_max_vel # the max distance the rover can possibly travel in a single direction
        self.y_bound = self.x_bound

        
    def init_states(self):
        '''
        Initialise discrete space
        '''
        # each possible position of the rover
        rover_possible_pos = list(itertools.product(np.arange(-self.x_bound, self.x_bound + 1), np.arange(-self.y_bound, self.y_bound + 1)))

        # each possible velocity of the rover
        rover_possible_vel = list(itertools.product(np.arange(-self.rover_max_vel, self.rover_max_vel + 1), np.arange(-self.rover_max_vel, self.rover_max_vel + 1)))

        # all combinations of each mineral being harvested/not harvested
        m_harvested_combinations = list(itertools.product(np.arange(0, 2), repeat=self.mineral_count))

        # each state is represented in the form ((rover_pos_x, rover_pos_y), (rover_vel_x, rover_vel_y), (m1_harvested, m2_harvested, ... ))
        states = list(itertools.product(rover_possible_pos, rover_possible_vel, m_harvested_combinations))
        disc_states = {}

        for i, _v in enumerate(states):
            disc_states[i] = _v
            
        return bidict(disc_states)

    def disc2state(self, s):
        s_def = self.disc_states[s]
        state = {}
        state['vel-x___d1'] = s_def[1][0]
        state['pos-x___d1'] = s_def[0][0]
        state['vel-y___d1'] = s_def[1][1]
        state['pos-y___d1'] = s_def[0][1]  
        for i in range(0, self.mineral_count):
            state[f'mineral-harvested___m{i + 1}'] = s_def[2][i]
        return state
    
    def state2disc(self, state):
        rover_pos_x = state['pos-x___d1']
        rover_pos_y = state['pos-y___d1']
        rover_pos = (rover_pos_x, rover_pos_y)
        rover_vel_x = state['vel-x___d1']
        rover_vel_y = state['vel-y___d1']
        rover_vel = (rover_vel_x, rover_vel_y)        
        mineral_harvest = []
        for i in range(1, self.mineral_count + 1):
            mineral_harvest.append(state[f'mineral-harvested___m{i}'])
        disc_state = rover_pos, rover_vel, tuple(mineral_harvest)
    
        if disc_state in self.disc_states.inverse:
            return self.disc_states.inverse[disc_state]
        else:
            index = len(self.disc_states)
            self.disc_states[index] = disc_state
            return index
    