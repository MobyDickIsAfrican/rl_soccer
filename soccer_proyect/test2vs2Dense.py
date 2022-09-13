
import numpy as np
import random
import matplotlib.pyplot as mpl
from env_2vs2 import Env2vs2
from plotter import generate_teams, generate_ball, get_angle
from spinup.utils.test_policy import load_policy_and_env
import matplotlib.pyplot as plt
import torch
from math import ceil
from TD3_team_alg_concat import TD3_team_alg

mpl.rcParams['font.size'] = 11
num_runs = 500
max_ep_len = 600
min_num = 8e6
RENDER = True




def main(rivals):
    teams=["home"]*2+["away"]*2
    exp_kwargs = {"free_play":False, "rivals": rivals, "actor_state_dict": rivals[random.randint(0, 2)]}
    env  = lambda : Env2vs2(team_1=2, team_2=2,task_kwargs={ "time_limit": 30, "disable_jump": True, 
                                "dist_thresh": 0.03, 'control_timestep': 0.1, "random_seed":69, "observables": "all"}) 

    exp_td3 = TD3_team_alg(env, 2, 2, epochs=300, max_ep_len=ceil(30 / 0.1), exp_kwargs=exp_kwargs)   
    fig, ax = plt.subplots()
    s = 0
    f = 0
    env = exp_td3.env
    pitch_size = env.dmcenv.task.arena._size
    
    for j in range(num_runs):
            
        obs = torch.Tensor(env.reset()).cuda()
        d = False
        l = 0
        while not(d or (l == max_ep_len)):
            step_position = [np.array(env.dmcenv._physics_proxy.bind(env.dmcenv.task.players[i].walker.root_body).xpos)[:2]
                               for i in range(4)]
            step_ball_pos = np.array(env.dmcenv._physics_proxy.bind(env.dmcenv.task.ball.root_body).xpos)[:2]
            obs_dict = env.timestep.observation[0]
            own_orientation = [obs_dict["sensors_gyro"][0, -1]]
            teammate_orientation = [get_angle(obs_dict[f'teammate_{i}_ego_orientation']) for i in range(1)]
            opp_orientation = [get_angle(obs_dict[f'opponent_{i}_ego_orientation']) for i in range(2)] 
            orientations = own_orientation+teammate_orientation+opp_orientation
            a = exp_td3.get_action(obs[np.newaxis, :], 0)
            a = [a[0,i, :] for i in range(4)]
            if RENDER:
                ax.cla()
                ax.set_xbound(-pitch_size[0], pitch_size[0])
                ax.set_ybound(-pitch_size[1], pitch_size[1])
                generate_teams(step_position, orientations, teams, ax)
                generate_ball(step_ball_pos)
                plt.pause(0.01)
                
                
            obs, r, d, _ = env.step(a)
            obs = torch.Tensor(obs).cuda()
            step_ball = env.get_ball()
            step_hit = step_ball.hit
            step_repossesed = step_ball.repossessed
            step_interception = step_ball.intercepted 
            step_dist = step_ball.dist_between_last_hits
            step_dist = step_dist if step_dist else 0
        done = 0
        if env.timestep.reward[0] > 0: 
            s += 1
            done = 1
        elif env.timestep.reward[0] <= 0: f += 1


import os
import json

base_path = "D:\\rl_soccer\\2vs0\\Junio_test_2022"
with open(os.path.join(base_path, "selected_models.json"), encoding='utf-8') as json_file:
    agents_json = json.load(json_file).values()
    agents = []
    for a_run in agents_json:
        agents+=a_run

rivals = [agents[22][0].split("\\"), agents[21][0].split("\\"), agents[19][0].split("\\")]
rivals = [os.path.join(base_path, *a_rival) for a_rival in rivals]
main(rivals)