from numpy.core import overrides
import dm_soccer2gym.wrapper as wrap
import numpy as np

class stage_soccerTraining(wrap.DmGoalWrapper):
	def __init__(self, team_1, team_2,task_kwargs={}, render_mode_list=None):
		super().__init__(team_1, team_2, task_kwargs, render_mode_list)
		# we define the rew_type for this particular stage 2 vs 0
		self.got_kickable_rew = 0
		self.current_team_with_ball = None
		self.old_ball_teammate_dist = []

	@overrides
	def set_vals(self):
		super().set_vals()
		self.got_kickable_rew = False
		
	def get_ball(self):
		'''
		This methods is created to get the ball from the environment, the ball 
		lets us know if the ball has been obtained by the same or different team. 
		Return: bool
		'''
		return self.dmcenv.ball

	def ball_intercepted(self):
		'''
		Funtion that returns True if the ball has been 
		intercepted by another team, and false otherwise.
		Return: bool
		'''
		ball = self.get_ball(self)
		return ball.intercepted()

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

	@overrides
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
		ball_teammate_pos = [-ball_pos[i] - obs[i]["teammate_0_ego_position"]
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
		kickable = np.any(ball_dist < self.dist_thresh)

		# rewards of each player in the game 
		alpha = (int(self.time_limit / self.control_timestep) + 1) / 10
		beta = alpha/10
		rewards = np.array(self.timestep.reward)

		# we check if there was a goal: 
		if not np.any(rewards):
			# there was not a goal: 
			kickable_now_first = (kickable * (1 - self.got_kickable_rew))
			# we set the distance of the goalpost and the ball. 
			delta_D = ((self.old_ball_op_goal_dist - ball_op_goal_dist) \
						- (self.old_ball_team_goal_dist - ball_team_goal_dist))
			delta_ball_d = ball_dist - self.old_ball_dist
			delta_teammate_ball_d = ball_teammate_dist - self.old_ball_teammate_dist 
			delta_ball_op_goal_dist = ball_op_goal_dist - self.old_ball_goal_dist
			
			kickable_reward = beta
			still_is_kickable_reward = 1.2*delta_D - np.max(delta_teammate_ball_d) - np.min(delta_ball_d) -0.1
			other_scenario_reward	= -np.min(delta_ball_d) -np.min(delta_ball_op_goal_dist) -0.1

			rewards += kickable_now_first*kickable_reward \
						+ (1- self.got_kickable_rew)*kickable*still_is_kickable_reward \
						+ (1-kickable)*other_scenario_reward

		self.old_ball_dist = ball_dist
		self.old_ball_op_dist = ball_op_goal_dist
		self.old_ball_team_goal_dist = ball_team_goal_dist
		self.old_ball_teammate_dist = ball_teammate_dist
		self.got_kickable_rew = kickable | self.got_kickable_rew

		return rewards.tolist()

	








	


	
