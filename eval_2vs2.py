
import argparse
from email.policy import default
from re import A
import dm_soccer2gym
import numpy as np
import os
import torch
from spinup.utils.test_policy import load_policy_and_env


num_runs = 500
max_ep_len = 600
RENDER = False


def main(team_1, team_2, model_path, order='home_away'):

    env = dm_soccer2gym.make("2vs2goal", task_kwargs={'rew_type': 'simple_v2', 'time_limit': 30., 'disable_jump': True, 'dist_thresh': .03, 
                             'control_timestep': 0.05, 'random_state': 69})
    team_1[0] = os.path.join(model_path, team_1[0])
    team_2[0] = os.path.join(model_path, team_2[0])
    
    if order == 'home_away':
        
        _, get_action_1 = load_policy_and_env(*team_1)
        
        _, get_action_2 = load_policy_and_env(*team_2)
    
    elif order == 'away_home':

        _, get_action_2 = load_policy_and_env(*team_1)
        
        _, get_action_1 = load_policy_and_env(*team_2)
    
    else:
        raise ValueError("Unexpected value for  parameter order, received %s, expected home_away | away_home" % (order))

    s = 0
    f = 0
    t = 0

    for j in range(num_runs):
            
        obs = torch.from_numpy(env.reset()[None, ...]).cuda()
        d = False
        l = 0

        while not(d or (l == max_ep_len)):
            a_home = get_action_1(obs[:, 0:2]).cpu()
            a_home = [a_home[0,i, :].numpy() for i in range(a_home.shape[2])]
            a_away = get_action_2(obs[:, 2:4]).cpu()
            a_away = [a_away[0,i, :].numpy() for i in range(a_away.shape[2])]
            a = [*a_home, *a_away]
            if RENDER:
                env.render()
            obs, r, d, _ = env.step(a)
            obs = torch.from_numpy(obs[None, ...]).cuda()
            l += 1

        if env.timestep.reward[0] > 0: s += 1
        elif env.timestep.reward[0] < 0: f += 1
        else: t += 1
            
        if (j + 1) % 10 == 0: 
            print(j + 1, s, t, f)

    return s, t, f


if __name__ == "__main__":
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents_path", type=str, help="File where agents model_path and num are", default="selected_models.csv")
    parser.add_argument("--model_path", type=str, help="path to where the models directories are stored", default="D:\\rl_soccer\\results\\resultados\\T1_2vs0_final")
    parser.add_argument("--team_1_idx", type=int, help="Agent 1 index", default=0)
    parser.add_argument("--team_2_idx", type=int, help="Agent 2 index", default=1)
    parser.add_argument("--save_path", type=str, help="Path to where save file with results", 
                        default="D:\\rl_soccer\\results\\resultados\\2vs0_final")
    parser.add_argument("--order", type=str, help="Order to be used for agents, home_away | away_home",
                        default='home_away')
    args = parser.parse_args()

    agents = pd.read_csv(args.agents_path).values[:, 1:]
    orig_path = os.getcwd()

    s, t, f = main(agents[args.team_1_idx], agents[args.team_2_idx], args.model_path, args.order)
    
    os.chdir(orig_path)
    os.makedirs(args.save_path, exist_ok=True)
        
    with open(os.path.join(args.save_path, f'{args.team_1_idx}vs{args.team_2_idx}.txt'), 'a') as f_:
        if args.order == 'home_away':
            f_.write("%d %d %d\n" % (s, t, f))
        else:
            f_.write("%d %d %d\n" % (f, t, s))

