from pyRDDLGym import RDDLEnv
from pyRDDLGym.Visualizer.MarsRoverViz import MarsRoverVisualizer
from pyRDDLGym.Core.Policies.Agents import RandomAgent

# set up the environment class, choose instance 0 because every example has at least one example instance
myEnv = RDDLEnv.RDDLEnv(domain="level 3/domain.rddl", instance="level 3/instance0.rddl")
# set up the environment visualizer
myEnv.set_visualizer(MarsRoverVisualizer)

agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.numConcurrentActions)

total_reward = 0
state = myEnv.reset()
for step in range(myEnv.horizon):
    myEnv.render()
    action = agent.sample_action()
    next_state, reward, done, info = myEnv.step(action)
    total_reward += reward
    print()
    print('step       = {}'.format(step))
    print('state      = {}'.format(state))
    print('action     = {}'.format(action))
    print('next state = {}'.format(next_state))
    print('reward     = {}'.format(reward))
    state = next_state
    if done:
        break
print("episode ended with reward {}".format(total_reward))
myEnv.close()
