import gym
from gym import spaces
	
class MarsGymEnv(gym.Env):
	"""Mars Environment that follows gym interface"""

	def __init__(self, arg1, arg2, ...):
		super(MarsGymEnv, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.action_space = spaces.Discrete(3)
		# Example for using image as input (channel-first; channel-last also works):
		self.observation_space = spaces.Box(low=0, high=255,
											shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

	def step(self, action):
		...
		return observation, reward, done, info
	
    def reset(self):
		self.done = False
		
		return observation  # reward, done, info can't be included