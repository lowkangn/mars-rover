import pickle

"""
Generates the transition model for a Mars Rover environment for Level 1.
"""
class TransitionModelGenerator(object):
    def __init__(self, env, instance='0'): 
        self.disc_states = env.disc_states.inverse
        self.disc_actions = env.disc_actions
        self.transitions = {}
        self.instance = instance
        
        self.x_bound = env.x_bound
        self.y_bound =  env.y_bound
        self.mineral_pos =  env.mineral_pos
        self.mineral_values =  env.mineral_values

        # cost functions (modified in accordance to environment)
        self.move_cost = 0.001
        self.harvest_cost = 1

    def generate_transitions(self):
        for s, s_i in self.disc_states.items():
            transitions_from_s = {}
            for a_i, a in self.disc_actions.items():
                
                action, value = a.split('|')
                p = 1.0 # probability is 1.0 as results of actions are deterministic
                s_j = s_i # index of next state
                r = 0 # reward

                if int(value) != 0: # if value is 0, then the rover is doing nothing
                    # x-movement
                    if action.startswith('move-x'):
                        rover_pos = s[0]
                        new_x = rover_pos[0] + int(value)
                        if abs(new_x) <= self.x_bound: # rover on edge of world boundary
                            s_ = ((new_x, rover_pos[1]), s[1])
                            s_j = self.disc_states[s_]
                        r = -self.move_cost
                        
                    # y-movement
                    elif action.startswith('move-y'):
                        rover_pos = s[0]
                        new_y = rover_pos[1] + int(value)
                        if abs(new_y) <= self.y_bound: # rover on edge of world boundary
                            s_ = ((rover_pos[0], new_y), s[1])
                            s_j = self.disc_states[s_]
                        r = -self.move_cost

                    # harvest
                    elif action.startswith('harvest'):
                        rover_pos = s[0]
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
                    else:
                        print('Unknown action encountered!')

                transitions_from_s[a_i] = p, s_j, r
            self.transitions[s_i] = transitions_from_s

            # save transition model for convenience
            with open(f'level 1/instance{self.instance}.pickle', 'wb') as f:
                pickle.dump(self.transitions, f)

        return self.transitions
