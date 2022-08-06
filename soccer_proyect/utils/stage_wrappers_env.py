import dm_soccer2gym.wrapper as wrap
from dm_soccer2gym.wrapper import polar_mod, polar_ang, sqrt_2, sigmoid
from collections import OrderedDict 
import numpy as np

def convertObservation(spec_obs):
	if not isinstance(spec_obs, list):
		if len(spec_obs.keys()) == 1:
			# no concatenation
			return list(spec_obs.values())[0]
		else:
			# concatentation
			numdim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs])
			space_obs = np.zeros((numdim,))
			i = 0
			for key in spec_obs:
				space_obs[i:i+np.int(np.prod(spec_obs[key].shape))] = spec_obs[key].ravel()
				i += int(np.prod(spec_obs[key].shape))
			return space_obs
	else:
		return [convertObservation(x) for x in spec_obs]

class stage_soccerTraining(wrap.DmGoalWrapper):
	def __init__(self, team_1, team_2,task_kwargs={}, render_mode_list=None):
		super().__init__(team_1, team_2, task_kwargs, render_mode_list)
		# we define the rew_type for this particular stage 2 vs 0
		
		self.current_team_with_ball = None
		

	def set_vals(self):
			obs = self.timestep.observation
			fl = self.timestep.observation[0]["field_front_left"][:, :2]
			br = self.timestep.observation[0]["field_back_right"][:, :2]

			self.max_dist = polar_mod(fl - br)

			self.got_kickable_rew = np.array([False for _ in range(self.num_players)])
			self.old_ball_dist = []
			self.old_ball_op_dist = []
			self.old_ball_team_dist = []
			self.old_ball_teammate_dist = []

			for i in range(self.num_players):
				ball_pos = -obs[i]['ball_ego_position'][:, :2]
				op_goal_pos = -obs[i]["opponent_goal_mid"][:, :2]
				tm_goal_pos = -obs[i]["team_goal_mid"][:, :2]
				ball_teammate_vec = -ball_pos - obs[i]["teammate_0_ego_position"][:,:2]

				ball_op_goal_pos = -ball_pos + op_goal_pos
				ball_team_goal_pos = -ball_pos + tm_goal_pos
				self.old_ball_dist.append(polar_mod(ball_pos) / self.max_dist)
				self.old_ball_op_dist.append(polar_mod(ball_op_goal_pos) / self.max_dist)
				self.old_ball_team_dist.append(polar_mod(ball_team_goal_pos) / self.max_dist)
				self.old_ball_teammate_dist.append(polar_mod(ball_teammate_vec)/self.max_dist)

			self.old_ball_dist = np.array(self.old_ball_dist)
			self.old_ball_op_goal_dist = np.array(self.old_ball_op_dist)
			self.old_ball_team_goal_dist = np.array(self.old_ball_team_dist)
			self.old_ball_teammate_dist = np.array(self.old_ball_teammate_dist)
			self.got_kickable_rew = np.abs(self.old_ball_dist)<self.dist_thresh
		
	def get_ball(self):
		'''
		This methods is created to get the ball from the environment, the ball 
		lets us know if the ball has been obtained by the same or different team. 
		Return: ball object 
		'''
		return self.dmcenv.task.ball

	def ball_intercepted(self):
		'''
		Funtion that returns True if the ball has been 
		intercepted by another team, and false otherwise.
		Return: bool
		'''
		ball = self.get_ball
		return ball.intercepted

	def ball_repossessed(self):
		'''
		Function that returns True if the same team that had the ball before
		has it now. Returns False otherwise.
		Return: bool 
		'''
		ball = self.get_ball()
		return ball.repossessed()


	def last_hit(self):
		'''
		Function that returns the player that hitted the ball last time. 
		Returns: Player named tuple
		'''
		return self.get_ball().last_hit()


	def getObservation(self):
			obs = self.timestep.observation
			cut_obs = []
			ball_pos_all = [-o['ball_ego_position'][:, :2] for o in obs]
			ball_dist_scaled_all = np.array([polar_mod(ball_pos) for ball_pos in ball_pos_all]) / self.max_dist
			kickable = np.abs(ball_dist_scaled_all) < self.dist_thresh
			kickable_ever = self.got_kickable_rew

			ctr = 0
			for i, o in enumerate(obs):
				ball_pos = ball_pos_all[ctr]
				ball_vel = o["ball_ego_linear_velocity"][:, :2]
				op_goal_pos = -o["opponent_goal_mid"][:, :2]
				team_goal_pos = -o["team_goal_mid"][:, :2]

				actual_vel = o["sensors_velocimeter"][:, :2]
				actual_ac = o["sensors_accelerometer"][:, :2]
				ball_op_goal_pos = -ball_pos + op_goal_pos
				ball_team_goal_pos = -ball_pos + team_goal_pos
				ball_goal_vel = o["stats_vel_ball_to_goal"]
				ball_dist_scaled = np.array([ball_dist_scaled_all[ctr]])

				cut_obs.append(OrderedDict({"ball_dist_scaled": ball_dist_scaled, 
											"ball_angle_scaled": np.array([polar_ang(ball_pos) / (2 * np.pi)]),
											"vel_norm_scaled": np.array([polar_mod(np.tanh(actual_vel)) / sqrt_2]), 
											"vel_ang_scaled": np.array([polar_ang(actual_vel) / (2 * np.pi)]),
											"ac_norm_scaled": np.array([polar_mod(np.tanh(actual_ac)) / sqrt_2]), 
											"ac_ang_scaled": np.array([polar_ang(actual_ac) / (2 * np.pi)]),
											"op_goal_dist_scaled": np.array([(polar_mod(op_goal_pos) / self.max_dist)]),
											"op_goal_angle_scaled": np.array([polar_ang(op_goal_pos) / (2 * np.pi)]),
											"team_goal_dist_scaled": np.array([(polar_mod(team_goal_pos) / self.max_dist)]),
											"team_goal_angle_scaled": np.array([polar_ang(team_goal_pos) / (2 * np.pi)]),
											"ball_vel_scaled": np.array([(polar_mod(np.tanh(ball_vel)) / sqrt_2)]), 
											"ball_vel_angle_scaled": np.array([polar_ang(ball_vel) / (2 * np.pi)]),
											"ball_op_goal_dist_scaled": np.array([(polar_mod(ball_op_goal_pos) / self.max_dist)]),
											"ball_op_goal_angle_scaled": np.array([polar_ang(ball_op_goal_pos) / (2 * np.pi)]),
											"ball_team_goal_dist_scaled": np.array([(polar_mod(ball_team_goal_pos) / self.max_dist)]),
											"ball_team_goal_angle_scaled": np.array([polar_ang(ball_team_goal_pos) / (2 * np.pi)]),
											"ball_goal_vel": np.array([sigmoid(ball_goal_vel)]),
											"kickable_ever": np.float32(kickable_ever[i])}))

				for player in range(self.team_2):
					opponent_pos = -o[f"opponent_{player}_ego_position"][:, :2]
					opponent_vel = o[f"opponent_{player}_ego_linear_velocity"][:, :2]
					opponent_ball_pos = -opponent_pos + ball_pos

					cut_obs[-1][f"opponent_{player}_dist_scaled"] = np.array([(polar_mod(opponent_pos) / self.max_dist)])
					cut_obs[-1][f"opponent_{player}_angle_scaled"] = np.array([(polar_ang(opponent_pos) / (2 * np.pi))])
					cut_obs[-1][f"opponent_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(opponent_vel)) / sqrt_2)])
					cut_obs[-1][f"opponent_{player}_vel_angle_scaled"] = np.array([(polar_ang(opponent_vel) / (2 * np.pi))])
					cut_obs[-1][f"opponent_{player}_ball_dist_scaled"] = np.array([(polar_mod(opponent_ball_pos) / self.max_dist)])
					cut_obs[-1][f"opponent_{player}_ball_angle_scaled"] = np.array([(polar_ang(opponent_ball_pos) / (2 * np.pi))])

				for player in range(self.team_1 - 1):
					teammate_pos = -o[f"teammate_{player}_ego_position"][:, :2]
					teammate_vel = o[f"teammate_{player}_ego_linear_velocity"][:, :2]
					teammate_ball_pos = -teammate_pos + ball_pos

					cut_obs[-1][f"teammate_{player}_dist_scaled"] = np.array([(polar_mod(teammate_pos) / self.max_dist)])
					cut_obs[-1][f"teammate_{player}_angle_scaled"] = np.array([(polar_ang(teammate_pos) / (2 * np.pi))])
					cut_obs[-1][f"teammate_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(teammate_vel)) / sqrt_2)])
					cut_obs[-1][f"teammate_{player}_vel_angle_scaled"] = np.array([(polar_ang(teammate_vel) / (2 * np.pi))])
					cut_obs[-1][f"teammate_{player}_ball_dist_scaled"] = np.array([(polar_mod(teammate_ball_pos) / self.max_dist)])
					cut_obs[-1][f"teammate_{player}_ball_angle_scaled"] = np.array([(polar_ang(teammate_ball_pos) / (2 * np.pi))])

				ctr += 1
			cutted_obs = convertObservation(cut_obs)
			return np.clip(cutted_obs, -1, 1)

	def calculate_rewards(self):
		# we get the observation: 
		obs = self.timestep.observation
		# we get the ball position:
		ball_pos = [-o['ball_ego_position'][:, :2] for o in obs]
		# we get the position of the ball and the goal post of the opposing team
		ball_op_goal_pos = [-ball_pos[i] - obs[i]["opponent_goal_mid"][:, :2] 
												for i in range(self.num_players)]
		# we get the position of the ball and the goal post of the team
		ball_team_goal_pos = [-ball_pos[i] - obs[i]["team_goal_mid"][:, :2] 
											for i in range(self.num_players)]
		# we get the teammate position with respect to the ball: 
		ball_teammate_pos = [-ball_pos[i] - obs[i]["teammate_0_ego_position"][:,:2]
										for i in range(self.num_players)]
		
		

		# we turn the ball position to polar normalized by the diagonal of the field
		ball_dist = np.array([wrap.polar_mod(ball_pos[i]) for i in range(self.num_players)]) / self.max_dist
		# we turn the opposing goal distance to the polar form normalized by the diagonal of the field 
		ball_op_goal_dist = np.array([wrap.polar_mod(ball_op_goal_pos[i]) for i in range(self.num_players)]) / self.max_dist
		# we turn the teams goal distance to the polar form, normalized by the diagonal
		ball_team_goal_dist = np.array([wrap.polar_mod(ball_team_goal_pos[i]) for i in range(self.num_players)]) / self.max_dist
		# ball teammate distance: 
		ball_teammate_dist = np.array([wrap.polar_mod(ball_teammate_pos[i])
															for i in range(self.num_players)])/self.max_dist
		

		# indicates if the ball is in a kickable position for any of both players
		kickable = np.abs(ball_dist) < self.dist_thresh

		# rewards of each player in the game 
		alpha = (int(self.time_limit / self.control_timestep) + 1)/10
		beta = alpha/10
		rewards = np.array(self.timestep.reward)

		# we check if there was a goal: 
		if not np.any(rewards):
			# there was not a goal: 
			kickable_now_first = kickable * (1 - self.got_kickable_rew)
			# we set the distance of the goalpost and the ball. 
			delta_D = ((self.old_ball_op_goal_dist - ball_op_goal_dist) \
						- (self.old_ball_team_goal_dist - ball_team_goal_dist))
			delta_ball_d = ball_dist - self.old_ball_dist
			delta_teammate_ball_d = ball_teammate_dist - self.old_ball_teammate_dist 
			delta_ball_op_goal_dist = ball_op_goal_dist - self.old_ball_op_goal_dist
			
			kickable_reward = beta
			still_is_kickable_reward =  (1.2*delta_D - np.min(delta_ball_d))[np.newaxis, ...]
			other_scenario_reward	= -np.min(delta_ball_d) -np.min(delta_ball_op_goal_dist) -0.1

			reward = kickable_now_first*kickable_reward \
						+ self.got_kickable_rew*kickable*still_is_kickable_reward \
						+ (1-kickable)*other_scenario_reward
			rewards += np.squeeze(reward, 0)
		
		else: 
			rewards *= alpha

		self.old_ball_dist = ball_dist
		self.old_ball_op_dist = ball_op_goal_dist
		self.old_ball_team_goal_dist = ball_team_goal_dist
		self.old_ball_teammate_dist = ball_teammate_dist
		self.got_kickable_rew = kickable #| self.got_kickable_rew

		return rewards.tolist()













	


	
