from dm_soccer2gym.wrapper import DmGoalWrapper
import numpy as np
from collections import OrderedDict
sigmoid = lambda x: 1 / (1 + np.exp(-x))
arctan_yx = lambda x, y: (np.arctan(np.divide(y, x) if x != 0 and y != 0 else 0) + np.pi * (x < 0)) % (2 * np.pi)
polar_mod = lambda x: np.sqrt(np.sum(np.square(x)))
polar_ang = lambda x: arctan_yx(x[0, 1], x[0, 0])
sqrt_2 = np.sqrt(2)

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
                space_obs[i:i+np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
                i += np.prod(spec_obs[key].shape)
            return space_obs
    else:
        return [convertObservation(x) for x in spec_obs]


class Env1vs0(DmGoalWrapper):
    
    def step(self, a):
        
        a_ = a.copy()
        if self.disable_jump:
            for j in range(len(a_)):
                a_[j] = np.concatenate([a_[j], np.array([0], dtype=np.float32)])
                
        self.timestep = self.dmcenv.step(a_)

        return self.getObservation(), self.calculate_rewards(), self.get_end(), {}

    def getObservation(self):
        obs = self.timestep.observation
        cut_obs = []
        ball_pos_all = [-o['ball_ego_position'][:, 1:] for o in obs]
        ball_dist_scaled_all = np.array([polar_mod(ball_pos) for ball_pos in ball_pos_all]) / self.max_dist
        kickable = ball_dist_scaled_all < self.dist_thresh
        kickable_ever = self.got_kickable_rew
        ctr = 0
        for o in obs:
            ball_pos = ball_pos_all[ctr]
            ball_vel = o["ball_ego_linear_velocity"][:, 1:]
            op_goal_pos = -o["opponent_goal_mid"][:, :2]
            team_goal_pos = -o["team_goal_mid"][:,:2]

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
                            		   "kickable_ever": np.float32(np.array([kickable_ever[ctr]]))}))

            for player in range(self.team_2):
                opponent_pos = -o[f"opponent_{player}_ego_position"][:, 1:]
                opponent_vel = o[f"opponent_{player}_ego_linear_velocity"][:, 1:]
                opponent_ball_pos = -opponent_pos + ball_pos

                cut_obs[-1][f"opponent_{player}_dist_scaled"] = np.array([(polar_mod(opponent_pos) / self.max_dist)])
                cut_obs[-1][f"opponent_{player}_angle_scaled"] = np.array([(polar_ang(opponent_pos) / (2 * np.pi))])
                cut_obs[-1][f"opponent_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(opponent_vel)) / sqrt_2)])
                cut_obs[-1][f"opponent_{player}_vel_angle_scaled"] = np.array([(polar_ang(opponent_vel) / (2 * np.pi))])
                cut_obs[-1][f"opponent_{player}_ball_dist_scaled"] = np.array([(polar_mod(opponent_ball_pos) / self.max_dist)])
                cut_obs[-1][f"opponent_{player}_ball_angle_scaled"] = np.array([(polar_ang(opponent_ball_pos) / (2 * np.pi))])

            for player in range(self.team_1 - 1):
                teammate_pos = -o[f"teammate_{player}_ego_position"][:, 1:]
                teammate_vel = o[f"teammate_{player}_ego_linear_velocity"][:, 1:]
                teammate_ball_pos = -teammate_pos + ball_pos

                cut_obs[-1][f"teammate_{player}_dist_scaled"] = np.array([(polar_mod(teammate_pos) / self.max_dist)])
                cut_obs[-1][f"teammate_{player}_angle_scaled"] = np.array([(polar_ang(teammate_pos) / (2 * np.pi))])
                cut_obs[-1][f"teammate_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(teammate_vel)) / sqrt_2)])
                cut_obs[-1][f"teammate_{player}_vel_angle_scaled"] = np.array([(polar_ang(teammate_vel) / (2 * np.pi))])
                cut_obs[-1][f"teammate_{player}_ball_dist_scaled"] = np.array([(polar_mod(teammate_ball_pos) / self.max_dist)])
                cut_obs[-1][f"teammate_{player}_ball_angle_scaled"] = np.array([(polar_ang(teammate_ball_pos) / (2 * np.pi))])

            ctr += 1

        o = np.clip(convertObservation(cut_obs), -1, 1)
        o = o.reshape(1, -1)
        return o, None