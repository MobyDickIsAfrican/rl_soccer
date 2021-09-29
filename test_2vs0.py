
import argparse
import csv
import dm_soccer2gym
import numpy as np
import os
from spinup.utils.test_policy import load_policy_and_env
from env_2vs0_pass import stage_soccerTraining_pass

import torch
num_runs = 500
max_ep_len = 600
min_num = 8e6
RENDER = True


def main(model_path, num, sess=None):
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
        vel1 = []
        vel2 = []
        pas1 = []
        pas2 = []

        while not(d or (l == max_ep_len)):
            a = get_action(obs[np.newaxis, :]).detach().cpu().numpy()
            a = [a[0,i, :] for i in range(2)]
            if RENDER:
                env.render()
            obs, r, d, _ = env.step(a)
            obs = torch.Tensor(obs).cuda()
            vel1.append(env.timestep.observation[0]['stats_vel_to_ball'])
            vel2.append(env.timestep.observation[1]['stats_vel_to_ball'])
            pas1.append(int(env.timestep.observation[0]['stats_i_received_pass']))
            pas2.append(int(env.timestep.observation[1]['stats_i_received_pass']))
            l += 1

        if env.timestep.reward[0] > 0: s += 1
        elif env.timestep.reward[0] <= 0: f += 1
        ls.append(l)
        vels1.append(np.mean(vel1))
        vels2.append(np.mean(vel2))
        passes1.append(sum(pas1))
        passes2.append(sum(pas2))
            
        if (j + 1) % 10 == 0: 
            total_pass = np.array(passes1)[-10:]+ np.array(passes2)[-10:]
            print(j + 1, s, f, np.mean(total_pass), np.max(total_pass), np.min(total_pass))

    return s, ls, vel1, vel2, passes1, passes2


if __name__ == "__main__":

    print("hi")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model path.")
    parser.add_argument("--meta_file", type=str, help="Name of meta file associated to checkpoint to load.")
    parser.add_argument("--gpu", type=float, help="Fraction of gpu to use", default=1.0)
    args = parser.parse_args()

    print("hi")

    model_path = args.model_path
    orig_path = os.getcwd()
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
    s, ls, vel1, vel2, pas1, pas2 = main(model_path,int(ckpoint_name))

    with open('stats.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.split(os.path.abspath(model_path))[1], str(ckpoint_name), str(s / num_runs), 
                         str(np.mean(ls)), str(np.std(ls)), str(np.mean(vel1)), str(np.std(vel1)),
                         str(np.mean(vel2)), str(np.std(vel2)), str(np.mean(pas1)), str(np.std(pas1)), str(np.mean(pas2)), str(np.std(pas2))])

    os.chdir(orig_path)
