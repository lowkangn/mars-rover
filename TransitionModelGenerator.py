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

    def move_x(self, s, s_i, value):
        s_j = s_i
        rover_pos = s[0]
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
    
    def harvest(self, s, s_i):
        pass
    
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

    def harvest(self, s, s_i):
        rover_pos = s[0]
        # harvests all possible minerals at once
        within = lambda m_pos, r: math.pow(m_pos[0] - rover_pos[0], 2) + math.pow(m_pos[1] - rover_pos[1], 2) <= math.pow(r, 2)
        to_mine = []
        r = -self.HARVEST_COST
        for i in range(self.mineral_count):
            # location check and not harvested before
            if s[1][i] == 0 and (rover_pos == self.mineral_pos[i] or within(self.mineral_pos[i], self.mineral_areas[i])):
                to_mine.append(i)
        if to_mine:        
            new_m = list(s[1])
            for m_i in to_mine:
                new_m[m_i] = 1
                r += self.mineral_values[m_i]
            new_m = tuple(new_m)
            s_ = (s[0], new_m)
            return self.disc_states[s_], r
        else:
            return s_i, r

class LevelThreeTransitionModel(TransitionModel):
    def __init__(self, env):
        super().__init__(env)

