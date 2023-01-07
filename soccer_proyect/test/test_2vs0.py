
import argparse
import csv
import dm_soccer2gym
import numpy as np
import os
import path
import sys
  
# directory reach
directory = path.path(__file__).abspath()
  
# setting path
sys.path.append(directory.parent.parent)

from spinup.utils.test_policy import load_policy_and_env
from env_2vs0_pass import stage_soccerTraining_pass


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
            
        obs = env.reset()
        d = False
        l = 0
        vel1 = []
        vel2 = []
        pas1 = []
        pas2 = []

        while not(d or (l == max_ep_len)):
            a = [get_action(obs[0])]
            if RENDER:
                env.render()
            obs, r, d, _ = env.step(a)
            vel1.append(env.timestep.observation[0]['stats_vel_to_ball'])
            vel2.append(env.timestep.observation[1]['stats_vel_to_ball'])
            pas1.append(env.timestep.observation[0]['stats_i_received_pass'])
            pas2.append(env.timestep.observation[1]['stats_i_received_pass'])
            l += 1

        if env.timestep.reward[0] > 0: s += 1
        elif env.timestep.reward[0] < 0: f += 1
        ls.append(l)
        vels1.append(np.mean(vel1))
        vels2.append(np.mean(vel2))
        passes1.append(pas1)
        passes2.append(pas2)
            
        if (j + 1) % 10 == 0: 
            print(j + 1, s, f)

    return s, ls, vel, 


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

    ckpoint_name = os.path.splitext(os.path.split(args.meta_file)[-1])[0]
    if len(ckpoint_name) > 11:
        num = int(ckpoint_name[12:])
    else:
        num = 'last'
    
    if type(num) != str and num < min_num:
        
        print(f"Skipping checkpoint number {num}")
    
    else:

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        s, ls, vels = main(model_path, num, sess)

        with open('stats.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([os.path.split(os.path.abspath(model_path))[1], str(num), str(s / num_runs), 
                             str(np.mean(ls)), str(np.std(ls)), str(np.mean(vels)), str(np.std(vels))])

        os.chdir(orig_path)
