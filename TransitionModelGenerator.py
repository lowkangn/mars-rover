import pickle
import math

"""
Generates the transition model for a Mars Rover environment.
"""
class TransitionModelGenerator(object):
    """
    factory
    """
    def __init__(self):
        self.lvldict = {'1': LevelOneTransitionModel, '2': LevelTwoTransitionModel, '3':LevelThreeTransitionModel}
    
    def get_tm(self, env):
        if env.level not in self.lvldict:
            raise Exception("no such level")
        return self.lvldict[env.level](env)

class TransitionModel(object):
    def __init__(self, env):
        self.level, self.instance = env.level, env.instance
        self.disc_states = env.disc_states.inverse
        self.disc_actions = env.disc_actions
        self.transitions = {}
        
        self.mineral_count = env.mineral_count
        self.mineral_pos = env.mineral_pos
        self.mineral_values = env.mineral_values
        self.mineral_areas = env.mineral_areas

        # cost functions (modified in accordance to environment)
        self.MOVE_COST = 0.001
        self.HARVEST_COST = 1

    def harvest(self, s, s_i):
        pass

    def action_generator(self, a, s, s_j, value):
        pass

    def save_tm(self):
        # save transition model for convenience
        with open(f'level {self.level}/instance{self.instance}.pickle', 'wb') as f:
            pickle.dump(self.transitions, f)
            f.close()
    
"""
For levels 1 and 2, where the rover's movement is based on displacement.
"""
class DisplacementTransitionModel(TransitionModel):
    def __init__(self, env):
        super().__init__(env)
        self.x_bound = env.x_bound
        self.y_bound = env.y_bound

    def movex(self, s, s_j, value):
        rover_pos = s[0]
        r = 0
        new_x = rover_pos[0] + int(value)
        if abs(new_x) <= self.x_bound: # rover on edge of world boundary
            s_ = ((new_x, rover_pos[1]), s[1])
            s_j = self.disc_states[s_]
        return s_j, -self.MOVE_COST
    
    def move_y(self, s, s_i, value):
        s_j = s_i
        rover_pos = s[0]
        new_y = rover_pos[1] + int(value)
        if abs(new_y) <= self.x_bound: # rover on edge of world boundary
            s_ = ((rover_pos[0], new_y), s[1])
            s_j = self.disc_states[s_]
        return s_j, -self.MOVE_COST

    def generate_transitions(self):
        for s, s_i in self.disc_states.items():
            transitions_from_s = {}
            for a_i, a in self.disc_actions.items():
                
                action, value = a.split('|')
                p = 1.0 # probability is 1.0 as results of actions are deterministic
                s_j = s_i # index of next state

                if int(value) != 0: # if value is 0, then the rover is doing nothing
                    # all possible actions
                    if action.startswith('move-x'): # x-movement
                        s_j, r = self.move_x(s, s_i, value)
                    elif action.startswith('move-y'): # y-movement
                        s_j, r = self.move_y(s, s_i, value)
                    elif action.startswith('harvest'): # harvest
                        s_j, r = self.harvest(s, s_i)
                    else:
                        print('Unknown action encountered!')

                transitions_from_s[a_i] = p, s_j, r
            self.transitions[s_i] = transitions_from_s

        self.save_tm()
        return self.transitions

class LevelOneTransitionModel(DisplacementTransitionModel):
    def __init__(self, env):
        super().__init__(env)
    
    def harvest(self, s, s_i):
        rover_pos = s[0]
        m_i = self.mineral_pos.index(rover_pos) if rover_pos in self.mineral_pos else -1

        if m_i >= 0 and s[1][m_i] == 0: # if rover is on mineral and mineral has not been harvested
            new_m = list(s[1])
            new_m[m_i] = 1
            new_m = tuple(new_m)
            s_ = (s[0], new_m)
            s_j = self.disc_states[s_]
            return s_j, self.mineral_values[m_i] - self.HARVEST_COST
        else:
            return s_i, -self.HARVEST_COST
    
class LevelTwoTransitionModel(DisplacementTransitionModel):
    def __init__(self, env):   
        super().__init__(env)

    def harvest(self, s, s_j):
        rover_pos = s[0]
        r = 0
        m_i = self.mineral_pos.index(rover_pos) if rover_pos in self.mineral_pos else -1
        if m_i >= 0 and s[1][m_i] == 0: # if rover is on mineral and mineral has not been harvested
            new_m = list(s[1])
            new_m[m_i] = 1
            new_m = tuple(new_m)
            s_ = (s[0], new_m)
            s_j = self.disc_states[s_]
            r = self.mineral_values[m_i] - self.harvest_cost
        else:
            r = -self.harvest_cost
        return s_j, r

    def action_generator(self, a, s, s_j, value):
        # all possible actions
        if a.startswith('move-x'): # x-movement
            s_j, r = self.movex(s, s_j, value)
        elif a.startswith('move-y'): # y-movement
            s_j, r = self.movey(s, s_j, value)
        elif a.startswith('harvest'): # harvest
            s_j, r = self.harvest(s, s_j)
        else:
            raise Exception('Unknown action encountered!')
        return s_j, r

class LevelThreeTransitionModel(TransitionModel):
    def __init__(self, env, level='3', instance='0'):
        super().__init__(env, level, instance)
        self.mineral_areas = env.mineral_areas
        self.rover_max_power = env.rover_max_power
        self.rover_max_vel = env.rover_max_vel
        self.move_cost = 0.1
    
    def power_x(self, s, s_j, value):
        rover_pos = s[0]
        rover_vel = s[1]
        r = 0
        new_vx = rover_vel[0] + int(value)
        new_x = rover_pos[0] + new_vx
        if abs(new_vx) <= self.rover_max_vel and abs(new_x) <= self.x_bound: # rover on edge of world boundary
            s_ = ((new_x, rover_pos[1]), (new_vx, rover_vel[1]), s[2])
            s_j = self.disc_states[s_]
            r -= int(value)/self.move_cost
        return s_j, r
    
    def power_y(self, s, s_j, value):
        rover_pos = s[0]
        rover_vel = s[1]
        r = 0
        new_vy = rover_vel[1] + int(value)
        new_y = rover_pos[1] + new_vy
        if abs(new_vy) <= self.rover_max_vel and abs(new_y) <= self.y_bound: # rover on edge of world boundary
            s_ = ((rover_pos[0], new_y), (rover_vel[0], new_vy), s[2])
            s_j = self.disc_states[s_]
            r -= int(value)/self.move_cost
        return s_j, r

    def harvest(self, s, s_j):
        rover_pos = s[0]
        r = 0
        m_i = self.mineral_pos.index(rover_pos) if rover_pos in self.mineral_pos else -1
        if m_i >= 0 and s[1][m_i] == 0: # if rover is on mineral and mineral has not been harvested
            new_m = list(s[1])
            new_m[m_i] = 1
            new_m = tuple(new_m)
            s_ = (s[0], s[1], new_m)
            s_j = self.disc_states[s_]
            r = self.mineral_values[m_i] - self.harvest_cost
        else:
            r = -self.harvest_cost
        return s_j, r

    def action_generator(self, a, s, s_j, value):
        # all possible actions
        if a.startswith('power-x'): # x-movement
            s_j, r = self.power_x(s, s_j, value)
        elif a.startswith('power-y'): # y-movement
            s_j, r = self.power_y(s, s_j, value)
        elif a.startswith('harvest'): # harvest
            s_j, r = self.harvest(s, s_j)
        else:
            raise Exception('Unknown action encountered!')
        return s_j, r

