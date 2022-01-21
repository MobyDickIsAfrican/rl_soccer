
import argparse
import csv
import dm_soccer2gym
import numpy as np
import os
from spinup.utils.test_policy import load_policy_and_env
from spinup.utils.logx import Logger, EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from env_2vs0_pass import stage_soccerTraining_pass
from dm_control.locomotion.soccer.team import Team, Player
from dm_soccer2gym.wrapper import polar_mod, polar_ang
from dm_control.locomotion.soccer.soccer_ball import SoccerBall
import math

import torch
num_runs = 500
max_ep_len = 600
min_num = 8e6
RENDER = True

def detect_hit_home_away(team_object):
    if team_object:
        if team_object.team.value ==0:
            goal_name = "HOME"+"_"+f"id={team_object.walker._walker_id}"
            home_or_away = 1
        else:
            goal_name = "AWAY"+"_"+f"id={team_object.walker._walker_id}"
            home_or_away = 0
        return home_or_away, goal_name
    else:
        return "N/A", -1


def detect_goal_home_away(team_object):
    if team_object:
        if team_object.value ==0:
            home_or_away = 1
        else:
            home_or_away = 0
        return home_or_away
    else:
        return -1

def main(model_path, num, m_file):
    m_name = m_file.split("/")[-1]
    instance_logger_kwargs = setup_logger_kwargs(f"{m_name}_instance_test", data_dir=model_path,datestamp=True)
    run_logger_kwargs = setup_logger_kwargs(f"{m_name}_test_run", data_dir=model_path,datestamp=True)

    # logger objects created. 
    run_logger = EpochLogger(**run_logger_kwargs)
    instance_logger = Logger(**instance_logger_kwargs)

    # Setup logger:
    run_logger.save_config(locals())
    instance_logger.save_config(locals())


    env  = stage_soccerTraining_pass(team_1=2, team_2=0,task_kwargs={ "time_limit": 30, "disable_jump": True, 
                                "dist_thresh": 0.03, 'control_timestep': 0.1, "random_seed":69, "observables": "all"}) 

    _, get_action = load_policy_and_env(model_path, num)

    s = 0
    f = 0
    ls = []
    vels1 = []
    vels2 = []
    passes1 = []
    passes2 = []

    for j in range(num_runs):
            
        obs = torch.Tensor(env.reset()).cuda()
        d = False
        l = 0
        reward = np.array([0]*2).astype('float32')
        vel = []
        passes = []
        interceptions = []
        repossessions = []
        hits = []
        distance_between_last_2_hits = []

        while not(d or (l == max_ep_len)):
            a = get_action(obs[np.newaxis, :]).detach().cpu().numpy()
            a = [a[0,i, :] for i in range(2)]
            if RENDER:
                env.render()
            obs, r, d, _ = env.step(a)
            obs = torch.Tensor(obs).cuda()
            step_ball = env.get_ball()
            step_hit = step_ball.hit
            step_repossesed = step_ball.repossessed
            step_interception = step_ball.intercepted 
            step_dist = step_ball.dist_between_last_hits
            step_dist = step_dist if step_dist else 0
            last_hitted = step_ball.last_hit
            hit_home_or_away, step_last_hit = detect_hit_home_away(last_hitted)
            goal_home_or_away = detect_goal_home_away(env.dmcenv.task.arena.detected_goal())
            ball_out = any([isinstance(instance, SoccerBall) for instance in env.dmcenv.task.arena.detected_off_court()] )

            step_position_x = [env.dmcenv._physics_proxy.bind(env.dmcenv.task.players[i].walker.root_body).xpos[0]
                               for i in range(2)]
            step_position_y = [env.dmcenv._physics_proxy.bind(env.dmcenv.task.players[i].walker.root_body).xpos[1]
                               for i in range(2)]


            step_ball_pos_x = env.dmcenv._physics_proxy.bind(env.dmcenv.task.ball.root_body).xpos[0]
            step_ball_pos_y = env.dmcenv._physics_proxy.bind(env.dmcenv.task.ball.root_body).xpos[1]

            step_pass = [player["stats_i_received_pass"][0] for player \
                                        in env.timestep.observation]
            passes1.append(step_pass[0])
            passes2.append(step_pass[1])
            step_vel = [player['stats_vel_to_ball'][0] for player \
                                            in env.timestep.observation]


            # log the information: 
            
            instance_logger.log_tabular("hit", step_hit)
            instance_logger.log_tabular("ball_position_x", step_ball_pos_x)
            instance_logger.log_tabular("ball_position_y", step_ball_pos_y)
            instance_logger.log_tabular("repossessed", step_repossesed)
            instance_logger.log_tabular("2_hit_dist", step_dist)
            instance_logger.log_tabular("interception", step_interception)
            instance_logger.log_tabular("last_hit", step_last_hit)
            [instance_logger.log_tabular(f"vel_{i}", velocity) for i, velocity in \
                                                enumerate(step_vel)]
            [instance_logger.log_tabular(f"received_pass_{i}", passing)\
                    for i, passing in enumerate(step_pass)]
            [instance_logger.log_tabular(f"player_{i}_pos_x", pos) for i, pos 
                                                in enumerate(step_position_x)]
            [instance_logger.log_tabular(f"player_{i}_pos_y", pos) for i, pos 
                                                in enumerate(step_position_y)]

            instance_logger.log_tabular("ball_out", ball_out)
            instance_logger.log_tabular("hit_H_A", hit_home_or_away)
            instance_logger.log_tabular("goal_H_A", goal_home_or_away)
            instance_logger.log_tabular("done", d)

            instance_logger.dump_tabular()


            vel_dict = {}
            pass_dict = {}
            for i, velocity in enumerate(step_vel):
                vel_dict[f"vel_player_{i}"] = velocity
                pass_dict[f"pass_player_{i}"] = step_pass[i]


            run_logger.store(**vel_dict)
            run_logger.store(**pass_dict)
            
            # add values to run information:
            vel.append(step_vel)
            interceptions.append(step_interception)
            repossessions.append(step_repossesed)
            hits.append(step_hit)
            distance_between_last_2_hits.append(step_dist)
            
            passes.append(step_pass)
            reward += np.array(r).astype('float32')
            l += 1

        done = 0
        if env.timestep.reward[0] > 0: 
            s += 1
            done = 1
        elif env.timestep.reward[0] <= 0: f += 1

        run_logger.log_tabular("steps", l)
        run_logger.log_tabular("success", done)
        run_logger.log_tabular("pitch_size", env.dmcenv.task.arena._size)
        [run_logger.log_tabular(f"vel_player_{i}") for i in range(2)]
        run_logger.dump_tabular()



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model path.", 
            default="D:\\rl_soccer\\results\\resultados\\2vs0_final\\2022-01-07_00-02-21_td3_soccer_goal_pass_join_2vs0_01_07_2022_00_02_21\\")
    parser.add_argument("--meta_file", type=str, help="Name of meta file associated to checkpoint to load.",
    default="D:\\rl_soccer\\results\\resultados\\2vs0_final\\2022-01-07_00-02-21_td3_soccer_goal_pass_join_2vs0_01_07_2022_00_02_21\\pyt_save\\best_models\\model19999.pt")
    parser.add_argument("--gpu", type=float, help="Fraction of gpu to use", default=1.0)
    args = parser.parse_args()

    model_path = args.model_path
    m_path = args.meta_file
    orig_path = os.getcwd()
    print(orig_path)
    os.chdir(model_path)

    ckpoint_name = os.path.splitext(os.path.split(args.meta_file)[-1])[0][5:] 
    print(ckpoint_name)
    '''
    if len(ckpoint_name) > 11:
        num = int(ckpoint_name[12:])
    else:
        num = 'last'

    if type(num) != str and num < min_num:
        
        print(f"Skipping checkpoint number {num}")

    else:
    '''
    main(model_path,int(ckpoint_name), m_path)
