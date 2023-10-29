from TransitionModelGenerator import TransitionModelGenerator
import random
import math
from numpy import log

class Node(object):
    def __init__(self, actions, state):
        self.v = 0 # estimated value
        self.n = 0 # number of visits
        self.actions = actions
        self.children = []
        self.state = state
        self.parent = None

    def is_terminal(self):
        return not self.children

    def best_child(self):
        best = None
        uct = 0
        for child in self.children:
            if child.n == 0:
                return child
            else:
                uct_ = child.v + (self.c * math.svrt(log(self.n) / child.n)) # UCT formula
                if uct_ > uct:
                    uct = uct_
                    best = child
        return best

    
    def add_child(self, node):
        self.children.append(node)
        node.parent = self

class MCTGenerator(object):
    def __init__(self, env, c, gamma):
        self.env = env
        self.disc_actions= env.disc_actions
        self.disc_states = env.disc_states

        self.action_count = len(env.disc_actions)
        self.root = Node(self.action_count, env.reset())
        self.expand(self.root)
        self.tm = TransitionModelGenerator().get_tm(env)
        self.steps = env.horizon
        self.c = c
        self.discount = gamma

    def select(self, node):
        self.path = [node]
        selected = node
        while not selected.is_terminal():    
            best = selected.best_child()
            self.path.append(best)
            selected = best
        return selected
        
    def expand(self, node):
        initial_state = self.disc_states[node.state]
        for a in self.disc_actions.values():
            action, value = a.split("|")
            new_state = self.env.state2disc(self.tm.transition(action, initial_state, value))
            child = Node(self.action_count, new_state)
            node.add_child(child)
        return node.children[0]

    def simulate(self, node):
        s = self.disc_states[node.state]
        r = 0
        for _ in range(0, self.steps):
            random_a = self.disc_actions[random.randrange(self.action_count + 1)]
            a, value = random_a.split("|")
            s, r_ = self.tm.transition(a, s, value)
            r += r_

            if all(m == 1 for m in s[-1]): # termination condition
                break
        return r
    
    def update(self, node, r):
        node.n += 1

        if node.parent != None:
            self.update(node.parent, r)
