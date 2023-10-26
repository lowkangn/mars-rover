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
        
        self.x_bound = env.x_bound
        self.y_bound = env.y_bound
        self.mineral_count = env.mineral_count
        self.mineral_pos = env.mineral_pos
        self.mineral_values = env.mineral_values

        # cost functions (modified in accordance to environment)
        self.HARVEST_COST = 1

    def harvest(self, s):
        pass

    def transition(self, a, s, value):
        pass

    def generate_transitions(self):
        for s, s_i in self.disc_states.items():
            transitions_from_s = {}
            for a_i, a in self.disc_actions.items():
                
                action, value = a.split('|')
                p = 1.0 # probability is 1.0 as results of actions are deterministic
                s_j = s_i # index of next state

                s_, r = self.transition(action, s, value)
                s_j = self.disc_states[s_]

                transitions_from_s[a_i] = p, s_j, r
            self.transitions[s_i] = transitions_from_s
            
        self.save_tm()
        return self.transitions

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

        # cost functions (modified in accordance to environment)
        self.MOVE_COST = 0.001

    def move_x(self, s, value):
        s_, r = s, 0
        rover_pos = s[0]
        new_x = rover_pos[0] + int(value)
        if abs(new_x) <= self.x_bound: # rover on edge of world boundary
            s_ = ((new_x, rover_pos[1]), s[1])
            r = -self.MOVE_COST
        return s_, r
    
    def move_y(self, s, value):
        s_, r = s, 0
        rover_pos = s[0]
        new_y = rover_pos[1] + int(value)
        if abs(new_y) <= self.x_bound: # rover on edge of world boundary
            s_ = ((rover_pos[0], new_y), s[1])
            r = -self.MOVE_COST
        return s_, r
    
    def transition(self, a, s, value):
        s_, r = s, 0
        if int(value) == 0: # if value is 0, then the rover is doing nothing
            return s_, r 
        # all possible actions
        if a.startswith('move-x'): # x-movement
            s_, r = self.move_x(s, value)
        elif a.startswith('move-y'): # y-movement
            s_, r = self.move_y(s, value)
        elif a.startswith('harvest'): # harvest
            s_, r = self.harvest(s)
        else:
            raise Exception('Unknown action encountered!')
        return s_, r
    
"""
For levels 2 and 3, where the minerals have a harvestable radius.
"""
class MineralRadiusTransitionModel(TransitionModel):
    def __init__(self, env):
        super().__init__(env)
        self.mineral_areas = env.mineral_areas

    def harvest(self, s):
        # harvests all possible minerals at once
        rover_pos = s[0]
        within = lambda m_pos, r: math.pow(m_pos[0] - rover_pos[0], 2) + math.pow(m_pos[1] - rover_pos[1], 2) <= math.pow(r, 2)
        to_mine = []
        s_, r = s, -self.HARVEST_COST
        for i in range(self.mineral_count):
            # location check and not harvested before
            if s[1][i] == 0 and (rover_pos == self.mineral_pos[i] or within(self.mineral_pos[i], self.mineral_areas[i])):
                to_mine.append(i)
        if to_mine:
            s_ = self.state_after_harvest(s, to_mine)
            r += sum(self.mineral_areas[m] for m in to_mine)
        return s_, r
    
    def state_after_harvest(self, s, to_mine):
        pass

class LevelOneTransitionModel(DisplacementTransitionModel):
    def __init__(self, env):
        super().__init__(env)
    
    def harvest(self, s):
        rover_pos = s[0]
        s_, r = s, -self.HARVEST_COST
        m_i = self.mineral_pos.index(rover_pos) if rover_pos in self.mineral_pos else -1

        if m_i >= 0 and s[1][m_i] == 0: # if rover is on mineral and mineral has not been harvested
            new_m = list(s[1])
            new_m[m_i] = 1
            new_m = tuple(new_m)
            s_ = (s[0], new_m)
            r += self.mineral_values[m_i]
        return s_, r
    
class LevelTwoTransitionModel(DisplacementTransitionModel, MineralRadiusTransitionModel):
    def __init__(self, env):   
        super().__init__(env)

    def state_after_harvest(self, s, to_mine):
        new_m = list(s[1])
        for m_i in to_mine:
            new_m[m_i] = 1
        new_m = tuple(new_m)
        return s[0], new_m

class LevelThreeTransitionModel(MineralRadiusTransitionModel):
    def __init__(self, env):
        super().__init__(env)
        self.rover_max_power = env.rover_max_power
        self.rover_max_vel = env.rover_max_vel

        # cost functions (modified in accordance to environment)
        self.POWER_COST = 0.1
    
    def power_x(self, s, value):
        rover_pos, rover_vel = s[0], s[1]
        s_, r = s, 0
        new_vx = max(-self.rover_max_vel, min(self.rover_max_vel, rover_vel[0] + int(value))) # velocity bounded by max vel
        s_ = (rover_pos, (new_vx, rover_vel[1]), s[2]) # new position will be calculated afterwards
        r = -int(value) / self.POWER_COST
        return s_, r
    
    def power_y(self, s, value):
        rover_pos, rover_vel = s[0], s[1]
        s_, r = s, 0
        new_vy = max(-self.rover_max_vel, min(self.rover_max_vel, rover_vel[0] + int(value))) # velocity bounded by max vel
        s_ = (rover_pos, (rover_vel[0], new_vy), s[2]) # new position will be calculated afterwards
        r = -int(value) / self.POWER_COST
        return s_, r
    
    def state_after_harvest(self, s, to_mine):
        new_m = list(s[2])
        for m_i in to_mine:
            new_m[m_i] = 1
        new_m = tuple(new_m)
        return s[0], s[1], new_m      

    def transition(self, a, s, value):
        # all possible actions, when value is 0, state can still change if rover is moving
        s_, r = s, 0
        if a.startswith('power-x'): # x-movement
            s_, r = self.power_x(s, value)
        elif a.startswith('power-y'): # y-movement
            s_, r = self.power_y(s, value)
        elif a.startswith('harvest'): # harvest
            if value != '0':
                s_, r = self.harvest(s)
        else:
            raise Exception('Unknown action encountered!')
        s_ = self.state_after_movement(s_) # calculate new position if rover's velocity is non-zero
        return s_, r
    
    def state_after_movement(self, s):
        rover_pos, rover_vel = s[0], s[1]
        new_x = max(-self.x_bound, min(self.x_bound, rover_pos[0] + rover_vel[0]))
        new_y = max(-self.y_bound, min(self.y_bound, rover_pos[1] + rover_vel[1]))
        return (new_x, new_y), rover_vel, s[2]
