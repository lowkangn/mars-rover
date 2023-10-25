import pickle
import math

"""
Generates the transition model for a Mars Rover environment.
"""
def TransitionModelGenerator(env, level, instance):
    """
    factory
    """
    lvldict = {'1': TransitionModelGenerator1, '2': TransitionModelGenerator2, '3':TransitionModelGenerator3}
    if level not in lvldict:
        raise Exception("no such level")
    return lvldict[level](env, level, instance)

class TransitionModelGenerator1(object):
    def __init__(self, env, level='1', instance='0'): 
        self.disc_states = env.disc_states.inverse
        self.disc_actions = env.disc_actions
        self.transitions = {}
        self.level = level
        self.instance = instance
        
        self.x_bound = env.x_bound
        self.y_bound =  env.y_bound
        self.mineral_count = env.mineral_count
        self.mineral_pos =  env.mineral_pos
        self.mineral_values =  env.mineral_values

        # cost functions (modified in accordance to environment)
        self.move_cost = 0.001
        self.harvest_cost = 1

    def movex(self, s, s_j, rover_pos, value):
        r = 0
        new_x = rover_pos[0] + int(value)
        if abs(new_x) <= self.x_bound: # rover on edge of world boundary
            s_ = ((new_x, rover_pos[1]), s[1])
            s_j = self.disc_states[s_]
            r = -self.move_cost
        return s_j, r
    
    def movey(self, s, s_j, rover_pos, value):
        r = 0
        new_y = rover_pos[1] + int(value)
        if abs(new_y) <= self.y_bound: # rover on edge of world boundary
            s_ = ((rover_pos[0], new_y), s[1])
            s_j = self.disc_states[s_]
            r = -self.move_cost
        return s_j, r

    def harvest(self, s, s_j, rover_pos):
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

    def generate_transitions(self):
        for s, s_i in self.disc_states.items():
            transitions_from_s = {}
            for a_i, a in self.disc_actions.items():
                
                action, value = a.split('|')
                p = 1.0 # probability is 1.0 as results of actions are deterministic
                s_j = s_i # index of next state

                if int(value) != 0: # if value is 0, then the rover is doing nothing
                    rover_pos = s[0]

                    # all possible actions
                    if action.startswith('move-x'): # x-movement
                        s_j, r = self.movex(s, s_j, rover_pos, value)
                    elif action.startswith('move-y'): # y-movement
                        s_j, r = self.movey(s, s_j, rover_pos, value)
                    elif action.startswith('harvest'): # harvest
                        s_j, r = self.harvest(s, s_j, rover_pos)
                    else:
                        print('Unknown action encountered!')

                transitions_from_s[a_i] = p, s_j, r
            self.transitions[s_i] = transitions_from_s

            # save transition model for convenience
            with open(f'level {self.level}/instance{self.instance}.pickle', 'wb') as f:
                pickle.dump(self.transitions, f)
                f.close()
        return self.transitions
    
class TransitionModelGenerator2(TransitionModelGenerator1):
    def __init__(self, env, level='2', instance='0'):   
        super().__init__(env, level, instance)
        self.mineral_areas = env.mineral_areas

    def harvest(self, s, s_j, rover_pos):
        # harvests all possible minerals at once
        within = lambda rover_pos, m_pos, r: math.pow(m_pos[0]-rover_pos[0], 2) + math.pow(m_pos[1]-rover_pos[1], 2) <= r**2
        to_mine = []
        r = -self.harvest_cost
        for i in range(self.mineral_count):
            # location check and not harvested before
            if s[1][i] == 0 and (rover_pos == self.mineral_pos[i] or within(rover_pos, self.mineral_pos[i], self.mineral_areas[i])):
                to_mine.append(i)
        if to_mine:        
            new_m = list(s[1])
            for m_i in to_mine:
                new_m[m_i] = 1
                r += self.mineral_values[m_i]
            new_m = tuple(new_m)
            s_ = (s[0], new_m)
            s_j = self.disc_states[s_]
        return s_j, r

class TransitionModelGenerator3(TransitionModelGenerator1):
    def __init__(self, env, level='3', instance='0'):
        super().__init__(env, level, instance)

