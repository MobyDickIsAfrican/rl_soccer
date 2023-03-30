from dm_soccer2gym.wrapper import polar_mod, polar_ang, sqrt_2, sigmoid
from collections import OrderedDict 
import numpy as np
from environments.env2vs2Dense import Env2vs2
from gym import core, spaces

class Env2vs0(Env2vs2):
	points =100
	def calculate_rewards(self):
		# we get the observation: 
		obs = self.timestep.observation
        # we find out if any of the players got the ball:
		gave_pass = np.squeeze(np.array([o['stats_i_received_pass_10m'] or o['stats_i_received_pass_15m'] for o in obs]))
		gave_pass = np.roll(gave_pass, 1)

		# rewards of each player in the game 
		alpha = (int(self.time_limit / self.control_timestep))
		beta = alpha/10
		rewards = np.array(self.timestep.reward)

		# we check if there was a goal: 
		if not np.any(rewards):
			# there was not a goal
			rewards += gave_pass*beta -(1-gave_pass)
		else: 
			rewards *= alpha
		return rewards.tolist()
