import dm_soccer2gym.wrapper as wrap
from dm_soccer2gym.wrapper import polar_mod, polar_ang, sqrt_2, sigmoid
from collections import OrderedDict 
import numpy as np
from gym import core, spaces
from torch import angle
from plotter import generate_ball, generate_teams, generate_text, to_degree
import matplotlib.pyplot as plt

def fix_angle(angle):
    if abs(angle)>np.pi:
        if angle<0:
            return 2*np.pi + angle
        else: 
            return -2*np.pi + angle
    else: 
        return angle

def convert_to_circle_angle(angle):
    if isinstance(angle, np.ndarray):
        angle = np.where(angle<0, angle+2*np.pi, angle)
    else:
        angle = angle+2*np.pi if angle<0 else angle
    return angle
def in_cone_fn(angle_range, thetaM):
    # two conditions angles have same sign or different sign:
    angle_range = list(map(fix_angle, angle_range))
    same_sign = np.prod(np.sign(angle_range)) > 0 
    if same_sign or (not same_sign)*(angle_range[0]<0):
        return  np.logical_and(thetaM>angle_range[0], thetaM<angle_range[1])
    else: 
        thetaM = convert_to_circle_angle(thetaM)
        angle_range_up_limit = convert_to_circle_angle(angle_range[1])
        return  np.logical_and(thetaM>angle_range[0], thetaM<angle_range_up_limit)

def convertObservation(spec_obs):
	if not isinstance(spec_obs, list):
		if len(spec_obs.keys()) == 1:
			# no concatenation
			return list(spec_obs.values())[0]
		else:
			# concatentation
			numdim = sum([int(np.prod(spec_obs[key].shape)) for key in spec_obs])
			space_obs = np.zeros((numdim,))
			i = 0
			for key in spec_obs:
				space_obs[i:i+int(np.prod(spec_obs[key].shape))] = spec_obs[key].ravel()
				i += int(np.prod(spec_obs[key].shape))
			return space_obs
	else:
		return [convertObservation(x) for x in spec_obs]

class Env2vs2(wrap.DmGoalWrapper):
    points =100
    def __init__(self, team_1, team_2,task_kwargs={}, render_mode_list=None):
        super().__init__(team_1, team_2, task_kwargs, render_mode_list)
        # we define the rew_type for this particular stage 2 vs 0
        self.cone_radiuos = 5
        self.cone_angle = np.pi/2
        self.cone_area = 0.5*self.cone_angle*(self.cone_radiuos)**2
        self.cone_pass_angle = np.pi/6
        self.cone_pass_area = lambda x: 0.5*(x**2)*self.cone_pass_angle
        self.angle_range_ego = (self.cone_angle-np.pi/4, self.cone_angle+np.pi/4)
        self.current_team_with_ball = None
        self.old_observation_space = self.observation_space
        self.observation_space = spaces.Box(0, 1, shape=(18 + (self.num_players - 1) * 7 + (team_1-1),))
        #fig, ax = plt.subplots()
        self.pitch_size = self.dmcenv.task.arena._size
        self.show = False
    def get_angle(self,rotation_matrix):
        rotation_matrix = rotation_matrix.reshape(3,3)
        phi = np.arctan2(rotation_matrix[1,0],rotation_matrix[0,0])
        return phi

    def get_Area(self, obs):
        # get the orientation of the agent
        own_orientation = obs["sensors_gyro"][0, -1]
        # we get the position of the teammate
        teammate_pos = [obs[f'teammate_0_ego_position'][0, :2].tolist()]
        # get the orientation of teammates players:
        teammate_orientation = [self.get_angle(obs[f'teammate_{i}_ego_orientation']) for i in range(self.team_1-1)]
        # get the orientation of each opponent
        opp_orientation = [self.get_angle(obs[f'opponent_{i}_ego_orientation']) for i in range(self.team_2)] 
        # get the position of each opponent:
        opp_pos = [obs[f'opponent_{i}_ego_position'][0, :2].tolist() for i in range(self.team_2)] 
        # concatenate observation:
        ego_other_player_pos = teammate_pos + opp_pos
        # concatenate orientations: 
        ego_other_player_angles =  teammate_orientation + opp_orientation 
        # get a list of intersections of each player with the agent:
        connect_areas = [self.calculate_intersection(own_orientation, angle_opp, pos[0], pos[1]) for angle_opp, pos in zip(ego_other_player_angles, ego_other_player_pos)]
        '''
        if self.now:
            ax = plt.gca()
            ax.cla()
            teams=["home"]*2+["away"]*2
            ax.set_xbound(-self.pitch_size[0], self.pitch_size[0])
            ax.set_ybound(-self.pitch_size[1], self.pitch_size[1])
            generate_teams([[0,0]]+ego_other_player_pos, [own_orientation]+ego_other_player_angles, teams, ax)
            generate_text(connect_areas, ax) 
            plt.waitforbuttonpress()
        
        '''
        
        connect_areas_teammate = {f"intercept_area_teammate_0": connect_areas.pop(0)}
        connect_areas_opponent = {f"intercept_area_opponent_{i}": connect_areas.pop(0)  for i in range(self.team_2)}

        ## GET THE PASS CONE INTERSECTION:
        pass_cone_area = 0
        centroid = None
        for team_pos, team_or in zip(teammate_pos, teammate_orientation):
            self.actual_pass_cone_area = self.cone_pass_area(polar_mod(team_pos))
            for angle_opp, pos in zip(opp_orientation, opp_pos):
                centroid_pos, area = self.calculate_intersection(team_or, angle_opp, pos[0], pos[1], pass_=True, team_pos = team_pos)
                if centroid is None:
                    centroid = centroid_pos
                pass_cone_area.append(area)
            
        pass_areas_teammate = {f"intercept_area_pass_teammate_{0}": sum(pass_cone_area),
                                f"intercept_centroid_pass_teammate_{0}": centroid_pos}

        cone_observation = {**connect_areas_teammate, **connect_areas_opponent, **pass_areas_teammate}
        return cone_observation
        

    
    def calculate_intersection(self, angle_ego, angle_opp, x_opp, y_opp, **pass_kwargs):
        # generate a angle range in which the cone should be:
        angle_range_ego  = [angle_ego-self.cone_angle/2, angle_ego+self.cone_angle/2]
        # define a delta radious for space discretization:
        dr = self.cone_radiuos/self.points
        # define a delta theta for space discretization:
        dtheta = self.cone_angle/self.points
        # generate a discrete vertor with the Radious of the cone:
        Rlim = np.linspace(0, self.cone_radiuos, num=self.points)
        # generate a range for the opponents cone
        angle_range_opp = [angle_opp-self.cone_angle/2, angle_opp+self.cone_angle/2]
        # generate a discrete vector for the Radious of the cone: 
        thetaopp = np.linspace(angle_range_opp[0], angle_range_opp[1], num=self.points)
        # get the area of intersection between the opponents cone and the agent
        intersection_area = self.get_area(Rlim, thetaopp, y_opp, x_opp, angle_range_ego, dr, dtheta, pass_kwargs)
        return intersection_area[0], intersection_area[-1]/self.cone_area

    def calculate_pass_cone_intersection(self, angle_ego, angle_opp, x_opp, y_opp):
        # get the teammate position
        team_pos = np.array(pass_kwargs.get("team_pos"))
        # get the proyected position of the half of the cone of the temmate
        half_proyected_pos = np.array([-0.5*self.cone_radiuos*np.sin(angle_ego), 0.5*self.cone_radiuos*np.cos(angle_ego)])
        # generate a proyected position
        proyected_pos = team_pos+half_proyected_pos
        # get the radious towards the proyected position
        position_radious = polar_mod(proyected_pos)
        # generate the angle for the proyected position
        angle_ego = polar_ang(proyected_pos[None, ...])
        # generate the range in which the angle of the cone moves:
        angle_range_ego  = [angle_ego - self.cone_pass_angle/2, angle_ego + self.cone_pass_angle/2]
        # generate a delta R for space discretization
        dr = position_radious/self.points
        # generate a delta Theta for space discretization:
        dtheta = (self.cone_pass_angle)/self.points
        # generate the pass key arguments for the intersection generation
        pass_kwargs = {"pass_": True,
                        "radious": position_radious,
                        "proyected_poss": proyected_pos,
                        "team_pos": team_pos,
                        "angle_ego": angle_ego}
        # vector of the discrete space for R:
        Rlim = np.linspace(0, self.cone_radiuos, num=self.points)
        # generate the cone for the opponent:
        angle_range_opp = [angle_opp-self.cone_angle/2, angle_opp+self.cone_angle/2]
        # vector for the discrete space of Theta:
        thetaopp = np.linspace(angle_range_opp[0], angle_range_opp[1], num=self.points)
        # calculate the intersection between this two areas:
        intersection_area = self.get_area(Rlim, thetaopp, y_opp, x_opp, angle_range_ego, dr, dtheta, centroid=False, **pass_kwargs)
        return [position_radious, angle_ego], np.array(intersection_area)/self.cone_pass_area
    
    def get_area(self, Rlim, thetaopp, y_opp, x_opp, angle_range_ego, dr, dtheta, has_centroid=True, **pass_kwargs):
        centroid = [0,0]
        if pass_kwargs.get("pass_", False):
            r = pass_kwargs.get("radious")
        else: 
            r = self.cone_radiuos
        Rv, thetav = np.meshgrid(Rlim, thetaopp)
        xval = -Rv*np.sin(thetav)
        yval = Rv*np.cos(thetav)
        thetaM = np.arctan2((yval+y_opp),(xval+x_opp))
        rM = np.sqrt(np.square(xval+x_opp)+np.square(yval+y_opp))
        min_r = np.min(rM)
        min_theta = np.min(thetaM)
        max_theta = np.max(thetaM)
        intersection_area = 0
        if min_theta<fix_angle(angle_range_ego[1]) and max_theta>fix_angle(angle_range_ego[0]) and min_r<r:
            thetaM_condition =in_cone_fn(angle_range_ego, thetaM)
            RM_high = np.where(rM<r, 1, 0)
            in_cone = thetaM_condition*RM_high
            intersection_area = np.sum(np.matmul(dr*dtheta*in_cone, Rlim.T))
            centroid = np.array([np.mean(RM_high), np.mean(thetaM)])
            '''
            if not self.pass_ and self.show:
                Rlim2 = np.linspace(0, r, num=self.points)
                thetaopp2 = np.linspace(angle_range_ego[0], angle_range_ego[1], num=self.points)
                Rv2, thetav2 = np.meshgrid(Rlim2, thetaopp2)
                xval2 = Rv2*np.cos(thetav2)
                yval2 = Rv2*np.sin(thetav2)
                Radious = np.sqrt(np.square(xval2)+np.square(yval2))
                fig, ax2 = plt.subplots()
                ax2.pcolor(xval2, yval2, Radious,alpha=0.3)
                ax2.pcolor(xval+x_opp, yval+y_opp, in_cone,alpha=0.7)
                plt.waitforbuttonpress()
            '''
        if has_centroid:
            return [centroid, intersection_area]
        else: 
            return [centroid, intersection_area]
        
        
        

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
        ball = self.get_ball(self)
        return ball.intercepted

    def ball_repossessed(self):
        '''
        Function that returns True if the same team that had the ball before
        has it now. Returns False otherwise.
        Return: bool 
        '''
        ball = self.get_ball()
        return ball.repossessed


    def last_hit(self):
        '''
        Function that returns the player that hitted the ball last time. 
        Returns: Player named tuple
        '''
        return self.get_ball().last_hit


    def getObservation(self):
            obs = self.timestep.observation
            cut_obs = []
            ball_pos_all = [-o['ball_ego_position'][:, :2] for o in obs]
            ball_dist_scaled_all = np.array([polar_mod(ball_pos) for ball_pos in ball_pos_all]) / self.max_dist
            kickable = np.abs(ball_dist_scaled_all) < self.dist_thresh
            kickable_ever = self.got_kickable_rew

            ctr = 0
            for i, o in enumerate(obs):
                self.now = i==0
                intercept_areas = self.get_Area(o)
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

                

                for player in range(self.team_1 - 1):
                    o_area = intercept_areas[f'intercept_area_teammate_{player}']
                    o_area_pass = intercept_areas[f"intercept_area_pass_teammate_{player}"]
                    o_centroid_pass = intercept_areas[f"intercept_centroid_pass_teammate_{player}"]
                    teammate_pos = -o[f"teammate_{player}_ego_position"][:, :2]
                    teammate_vel = o[f"teammate_{player}_ego_linear_velocity"][:, :2]
                    teammate_ball_pos = -teammate_pos + ball_pos

                    cut_obs[-1][f"teammate_{player}_dist_scaled"] = np.array([(polar_mod(teammate_pos) / self.max_dist)])
                    cut_obs[-1][f"teammate_{player}_angle_scaled"] = np.array([(polar_ang(teammate_pos) / (2 * np.pi))])
                    cut_obs[-1][f"teammate_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(teammate_vel)) / sqrt_2)])
                    cut_obs[-1][f"teammate_{player}_vel_angle_scaled"] = np.array([(polar_ang(teammate_vel) / (2 * np.pi))])
                    cut_obs[-1][f"teammate_{player}_ball_dist_scaled"] = np.array([(polar_mod(teammate_ball_pos) / self.max_dist)])
                    cut_obs[-1][f"teammate_{player}_ball_angle_scaled"] = np.array([(polar_ang(teammate_ball_pos) / (2 * np.pi))])
                    cut_obs[-1][f"teammate_{player}_centroid_area_R"] = np.array(o_area[0][0]/ self.max_dist)
                    cut_obs[-1][f"teammate_{player}_centroid_area_theta"] = np.array(o_area[0][1]/ (2 * np.pi))
                    cut_obs[-1][f"teammate_{player}_intercept_area"] = np.array(o_area[1])
                    cut_obs[-1][f"teammate_{player}_centroid_pass_R"] = np.array(o_area_pass/ self.max_dist)
                    cut_obs[-1][f"teammate_{player}_centroid_pass_theta"] = np.array(o_centroid_pass[0]/ (2 * np.pi))
                    cut_obs[-1][f"teammate_{player}_intercept_pass"] = np.array(o_centroid_pass[1])
                
                for player in range(self.team_2):
                    o_area = intercept_areas[f'intercept_area_opponent_{player}']
                    opponent_pos = -o[f"opponent_{player}_ego_position"][:, :2]
                    opponent_vel = o[f"opponent_{player}_ego_linear_velocity"][:, :2]
                    opponent_ball_pos = -opponent_pos + ball_pos

                    cut_obs[-1][f"opponent_{player}_dist_scaled"] = np.array([(polar_mod(opponent_pos) / self.max_dist)])
                    cut_obs[-1][f"opponent_{player}_angle_scaled"] = np.array([(polar_ang(opponent_pos) / (2 * np.pi))])
                    cut_obs[-1][f"opponent_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(opponent_vel)) / sqrt_2)])
                    cut_obs[-1][f"opponent_{player}_vel_angle_scaled"] = np.array([(polar_ang(opponent_vel) / (2 * np.pi))])
                    cut_obs[-1][f"opponent_{player}_ball_dist_scaled"] = np.array([(polar_mod(opponent_ball_pos) / self.max_dist)])
                    cut_obs[-1][f"opponent_{player}_ball_angle_scaled"] = np.array([(polar_ang(opponent_ball_pos) / (2 * np.pi))])
                    cut_obs[-1][f"opponent_{player}_centroid_area_R"] = np.array(o_area[0][0]/ self.max_dist)
                    cut_obs[-1][f"opponent_{player}_centroid_area_theta"] = np.array(o_area[0][1]/ (2 * np.pi))
                    cut_obs[-1][f"opponent_{player}_intercept_area"] = np.array(o_area[1])
                ctr += 1
            cutted_obs = convertObservation(cut_obs)
            return np.clip(cutted_obs, -1, 1)
            
    def calculate_rewards(self):
        # we get the observation: 
        obs = self.timestep.observation
        # we find out if any of the players got the ball:
        gave_pass = np.squeeze(np.array([o['stats_i_received_pass_10m'] or o['stats_i_received_pass_15m'] for o in obs]))
        gave_pass = np.array([any(gave_pass[:self.team_1])]*self.team_1 + [any(gave_pass[self.team_1:self.team_1 + self.team_2])]*self.team_2).astype(int)
        opponent_intercepted = np.squeeze(np.array([obs[0]["stats_opponent_intercepted_ball"]]*self.team_1 + [obs[-1]["stats_opponent_intercepted_ball"]]*self.team_2), -1)


        # rewards of each player in the game 
        alpha = (int(self.time_limit / self.control_timestep))
        beta = alpha/10
        beta_intercept = beta/10
        rewards = np.array(self.timestep.reward)

        # we check if there was a goal: 
        if not np.any(rewards):
        # there was not a goal
            positive_reward = gave_pass*beta + beta_intercept*np.logical_not(opponent_intercepted)*np.any(opponent_intercepted)
            punish_reward = - beta_intercept*opponent_intercepted
            otherwise_reward = -1*(1 - np.logical_or(positive_reward!=0, punish_reward!=0))
            rewards +=   positive_reward + punish_reward + otherwise_reward
        else: 
            rewards *= alpha
        return rewards.tolist()
