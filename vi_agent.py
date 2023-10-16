from MarsRoverDisc import MarsRoverDisc

myEnv = MarsRoverDisc()

# agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.numConcurrentActions)

# total_reward = 0
# state = myEnv.reset()
# for step in range(myEnv.horizon):
#     myEnv.render()
#     action = agent.sample_action()
#     next_state, reward, done, info = myEnv.step(action)
#     total_reward += reward
#     print()
#     print('step       = {}'.format(step))
#     print('state      = {}'.format(state))
#     print('action     = {}'.format(action))
#     print('next state = {}'.format(next_state))
#     print('reward     = {}'.format(reward))
#     state = next_state
#     if done:
#         break
# print("episode ended with reward {}".format(total_reward))
# myEnv.close()
