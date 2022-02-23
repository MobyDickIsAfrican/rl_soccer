
import argparse
import csv
import dm_soccer2gym
import numpy as np
import os
from spinup.utils.test_policy import load_policy_and_env
import tensorflow as tf


test_agent = ("/home/amtc/pavan/rl_soccer/models/TD3/paper/1vs0/2020-09-12_23-40-25_td3_soccer_goal_1vs0_simple_v2_0.05", 8299999)
num_runs = 100
max_ep_len = 600
RENDER = False


def main(model_path, num, sess=None, order="home_away"):
    env = dm_soccer2gym.make("1vs1goal", task_kwargs={'rew_type': 'simple_v2', 'time_limit': 30., 'disable_jump': True, 'dist_thresh': .03, 
                             'control_timestep': 0.05, 'random_state': 42})
    
    if order == 'home_away':
    
        _, get_action_1 = load_policy_and_env(model_path, num, sess=sess)
        
        g2 = tf.Graph()
        
        with g2.as_default():
            _, get_action_2_ = load_policy_and_env(*test_agent, sess=None)
            get_action_2 = lambda x: get_action_2_(x[:18])
        
        idx = 0
    
    elif order == 'away_home':
    
        _, get_action_1_ = load_policy_and_env(*test_agent, sess=sess)
        get_action_1 = lambda x: get_action_1_(x[:18])
        
        g2 = tf.Graph()
        
        with g2.as_default():
            _, get_action_2 = load_policy_and_env(model_path, num, sess=None)
        
        idx = 1
        
    else:
        raise ValueError

    s = 0
    f = 0
    ls = []
    vels = []

    for j in range(num_runs):
            
        obs = env.reset()
        d = False
        l = 0
        vel = []

        while not(d or (l == max_ep_len)):
            a = [get_action_1(obs[0]), get_action_2(obs[1])]
            if RENDER:
                env.render()
            obs, r, d, _ = env.step(a)
            vel.append(env.timestep.observation[idx]['stats_vel_to_ball'])
            l += 1

        if env.timestep.reward[idx] > 0: s += 1
        elif env.timestep.reward[idx] < 0: f += 1
        ls.append(l)
        vels.append(np.mean(vel))
            
        if (j + 1) % 10 == 0: 
            print(j + 1, s, f)

    return s, f, ls, vels


if __name__ == "__main__":

    print("hi")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model path.")
    parser.add_argument("--meta_file", type=str, help="Name of meta file associated to checkpoint to load.")
    parser.add_argument("--gpu", type=float, help="Fraction of gpu to use", default=1.0)
    parser.add_argument("--order", type=str, help="Order to be used for loaded and test agent, home_away | away_home",
                        default='home_away')
    parser.add_argument("--save_path", type=str, help="Path in which to save files, if none use model_path", default="")
    args = parser.parse_args()

    model_path = args.model_path
    orig_path = os.getcwd()
    os.chdir(os.path.join(model_path, ".."))

    ckpoint_name = os.path.splitext(os.path.split(args.meta_file)[-1])[0]
    if len(ckpoint_name) <= 11:
        print(f"Skipping checkpoint {meta_file}")
        

    else:
        num = int(ckpoint_name[12:])

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        s, f, ls, vels = main(model_path, num, sess, args.order)
            
        os.chdir(orig_path)
            
        save_path = args.save_path if len(args.save_path) > 0 else model_path
            
        os.makedirs(os.path.join(save_path, 'stats'), exist_ok=True)
        print(os.path.join(save_path, 'stats'))
            
        main_name = os.path.split(os.path.abspath(model_path))[1]
        with open(os.path.join(save_path, f'stats/{main_name}_{num}.txt'), 'a+') as _file:
            writer = csv.writer(_file)
            writer.writerow([main_name, str(num), str(s / num_runs), 
                             str(f / num_runs), str(np.mean(ls)), str(np.std(ls)), str(np.mean(vels)), 
                             str(np.std(vels))])
