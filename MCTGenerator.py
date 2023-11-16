from TransitionModelGenerator import TransitionModelGenerator
import random
import math
from numpy import log

class Node(object):
    def __init__(self, action_count, action):
        self.q = 0 # estimated value
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
                uct_ = (child.q / child.n) + (self.c * math.sqrt(log(self.n) / child.n)) # UCT formula
                if uct_ > uct:
                    uct = uct_
                    best = child
        return best
    
    def add_child(self, node):
        self.children.append(node)
        self.unexplored.remove(node.action)
        node.parent = self

class MCTGenerator(object):
    def __init__(self, env, gamma, max_depth):
        self.env = env
        self.gamma = gamma
        self.max_depth = max_depth

        self.action_count = len(env.disc_actions)
        self.root = self.reset()
        self.horizon = env.horizon

        # Store the sequence of actions found by MCTS for running the simulation later.
		# Since the environment is deterministic, the sequence of steps can be predetermined.
        self.steps = []
        self.x_list = []
        self.y_list = []

    def reset(self):
        self.root = Node(self.action_count, None)
        return self.root

    def next_action(self):
        action = max(self.root.children, key=lambda child: child.q / child.n).action
        self.steps.append(action)
        self.reset()
        return action

    def select(self):
        self.env.reset()
        for action in self.steps:
            self.env.step(action)

        selected = self.root
        while selected.is_fully_expanded():  
            best = selected.best_child()
            selected = best
            self.env.step(selected.action)
        return selected
        
    def expand(self, node):
        a = random.sample(list(node.unexplored), 1)[0]
        child = Node(self.action_count, a)
        node.add_child(child)
        s_, r, _, _ = self.env.step(a)
        rover_pos = self.env.disc_states[s_][0]
        self.x_list.append(rover_pos[0])
        self.y_list.append(rover_pos[1])
        return child, r

    def simulate(self, initial_r):
        r = initial_r
        discount = 1
        for _ in range(min(self.max_depth, self.horizon - len(self.steps))):
            random_a = random.randrange(self.action_count)
            s_, reward, done, _ = self.env.step(random_a)
            r += (reward * discount)
            discount *= self.gamma
            rover_pos = self.env.disc_states[s_][0]
            self.x_list.append(rover_pos[0])
            self.y_list.append(rover_pos[1])
            if done:
                break
        return r
    
    def update(self, node, r): # backpropagation    
        node.n += 1
        # node.q = ((node.q * node.n) + (r * new_n)) / (node.n + new_n)
        node.q += r

        if node.parent != None:
            self.update(node.parent, r * self.gamma)
