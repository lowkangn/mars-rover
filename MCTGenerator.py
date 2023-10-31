from TransitionModelGenerator import TransitionModelGenerator
import random
import math
from numpy import log

class Node(object):
    def __init__(self, action_count, action):
        self.v = 0 # estimated value
        self.n = 0 # number of visits
        self.c = 2 # exploration parameter

        self.action_count = action_count
        self.action = action
        self.children = []
        self.parent = None

    def is_terminal(self):
        return not self.children

    def best_child(self):
        best = None
        uct = -math.inf
        for child in self.children:
            if child.n == 0:
                return child
            else:
                uct_ = child.v + (self.c * math.sqrt(log(self.n) / child.n)) # UCT formula
                if uct_ > uct:
                    uct = uct_
                    best = child
        return best
    
    def select_random(self):
        return random.choice(self.children)
    
    def add_child(self, node):
        self.children.append(node)
        node.parent = self

class MCTGenerator(object):
    def __init__(self, env):
        self.env = env

        self.action_count = len(env.disc_actions)
        self.root = self.reset()
        self.horizon = env.horizon

        # Store the sequence of actions found by MCTS for running the simulation later.
		# Since the environment is deterministic, the sequence of steps can be predetermined.
        self.steps = []

    def reset(self):
        self.root = Node(self.action_count, None)
        self.expand(self.root)
        return self.root

    def next_action(self):
        action = max(self.root.children, key=lambda child: child.v).action
        self.steps.append(action)
        return action

    def select(self):
        self.env.reset()
        for action in self.steps:
            self.env.step(action)

        selected = self.root
        while not selected.is_terminal():
            best = selected.best_child()
            selected = best
        return selected
        
    def expand(self, node):
        for a in range(self.action_count):
            child = Node(self.action_count, a)
            node.add_child(child)
        return node.children[0]

    def simulate(self):
        r = 0
        for _ in range(0, self.horizon - len(self.steps)):
            random_a = random.randrange(self.action_count)
            _, reward, done, _ = self.env.step(random_a)
            r += reward
            if done:
                break
        return r
    
    def update(self, node, r): # backpropagation
        node.n += 1
        node.v += r

        if node.parent != None:
            self.update(node.parent, r)
