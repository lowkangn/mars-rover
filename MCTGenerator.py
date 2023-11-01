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
        self.unexplored = set(range(action_count))
        self.children = []
        self.parent = None

    def is_fully_expanded(self):
        return not self.unexplored

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
    
    def add_child(self, node):
        self.children.append(node)
        self.unexplored.remove(node.action)
        node.parent = self

class MCTGenerator(object):
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma

        self.action_count = len(env.disc_actions)
        self.root = self.reset()
        self.horizon = env.horizon

        # Store the sequence of actions found by MCTS for running the simulation later.
		# Since the environment is deterministic, the sequence of steps can be predetermined.
        self.steps = []

    def reset(self):
        self.root = Node(self.action_count, None)
        for _ in range(self.action_count):
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
        while selected.is_fully_expanded():
            best = selected.best_child()
            selected = best
        return selected
        
    def expand(self, node):
        a = random.sample(list(node.unexplored), 1)[0]
        child = Node(self.action_count, a)
        node.add_child(child)
        return child

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
        new_n = node.n + 1
        node.v = ((node.v * node.n) + (r * new_n)) / (node.n + new_n)
        node.n = new_n

        if node.parent != None:
            self.update(node.parent, r * self.gamma)
