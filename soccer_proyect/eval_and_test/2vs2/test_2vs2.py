
import argparse
import csv
import dm_soccer2gym
import numpy as np
import os
from spinup.utils.test_policy import load_policy_and_env
import tensorflow as tf


test_agent = [("/home/amtc/pavan/rl_soccer/models/TD3/paper/1vs1/2020-10-08_23-06-32_td3_soccer_goal_1vs1_simple_v2_0.05", 9389999),
              ("/home/amtc/pavan/rl_soccer/models/TD3/paper/1vs1/2020-10-08_23-07-33_td3_soccer_goal_1vs1_simple_v2_0.05", 8629999)]
num_runs = 100
max_ep_len = 900
RENDER = False


def main(model_path, num, sess=None, order="home_away"):
    env = dm_soccer2gym.make("2vs2goal", task_kwargs={'rew_type': 'simple_v2', 'time_limit': 45., 'disable_jump': True, 'dist_thresh': .03, 
                             'control_timestep': 0.05, 'random_state': 42})
    
    if order == 'home_away':
    
        _, get_action_1_2 = load_policy_and_env(model_path, num, sess=sess, two_p=True)
        
        g2 = tf.Graph()
        
        with g2.as_default():
            _, get_action_3_ = load_policy_and_env(*test_agent[0], sess=None)
            get_action_3 = lambda x: get_action_3_(x[np.r_[0:18, 18:24]])
            get_action_3_r = lambda x: get_action_3_(x[np.r_[0:18, 24:30]])

        g3 = tf.Graph()
        
        with g3.as_default():
            _, get_action_4_ = load_policy_and_env(*test_agent[1], sess=None)
            get_action_4 = lambda x: get_action_4_(x[np.r_[0:18, 24:30]])
            get_action_4_r = lambda x: get_action_4_(x[np.r_[0:18, 18:24]])
        
        idx = [0, 1]
    
    elif order == 'away_home':
    
        _, get_action_1_ = load_policy_and_env(*test_agent[0], sess=sess)
        get_action_1 = lambda x: get_action_1_(x[np.r_[0:18, 18:24]])
        get_action_1_r = lambda x: get_action_1_(x[np.r_[0:18, 24:30]])
        
        g2 = tf.Graph()
        
        with g2.as_default():
            _, get_action_2_ = load_policy_and_env(*test_agent[1], sess=None)
            get_action_2 = lambda x: get_action_2_(x[np.r_[0:18, 24:30]])
            get_action_2_r = lambda x: get_action_2_(x[np.r_[0:18, 18:24]])

        g3 = tf.Graph()
        
        with g3.as_default():
            _, get_action_3_4 = load_policy_and_env(model_path, num, sess=None, two_p=True)
        
        idx = [2, 3]
        
    else:
        raise ValueError

    s = 0
    f = 0
    ls = []
    vels_1 = []
    vels_2 = []

    for j in range(num_runs):

        if order == 'home_away':

            if j % 2 == 0:
                get_action_3_4 = lambda x, y: (get_action_3(x), get_action_4(y))

            else:
                get_action_3_4 = lambda x, y: (get_action_4_r(x), get_action_3_r(y)                )

        else:

            if j % 2 == 0:
                get_action_1_2 = lambda x, y: (get_action_1(x), get_action_2(y))

            else:
                get_action_1_2 = lambda x, y: (get_action_2_r(x), get_action_1_r(y))
            
        obs = env.reset()
        d = False
        l = 0
        vel_1 = []
        vel_2 = []

        while not(d or (l == max_ep_len)):
            a = [*get_action_1_2(obs[0], obs[1]), *get_action_3_4(obs[2], obs[3])]
            if RENDER:
                env.render()
            obs, r, d, _ = env.step(a)
            vel_1.append(env.timestep.observation[idx[0]]['stats_vel_to_ball'])
            vel_2.append(env.timestep.observation[idx[1]]['stats_vel_to_ball'])
            l += 1

        if env.timestep.reward[idx[0]] > 0: s += 1
        elif env.timestep.reward[idx[0]] < 0: f += 1
        ls.append(l)
        vels_1.append(np.mean(vel_1))
        vels_2.append(np.mean(vel_2))
            
        if (j + 1) % 10 == 0: 
            print(j + 1, s, f)

    return s, f, ls, vels_1, vels_2


if __name__ == "__main__":

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
        s, f, ls, vels_1, vels_2 = main(model_path, num, sess, args.order)
            
        os.chdir(orig_path)
            
        save_path = args.save_path if len(args.save_path) > 0 else model_path
            
        os.makedirs(os.path.join(save_path, 'stats'), exist_ok=True)
        print(os.path.join(save_path, 'stats'))
            
        main_name = os.path.split(os.path.abspath(model_path))[1]
        with open(os.path.join(save_path, f'stats/{main_name}_{num}.txt'), 'a+') as _file:
            writer = csv.writer(_file)
            writer.writerow([main_name, str(num), str(s / num_runs), 
                             str(f / num_runs), str(np.mean(ls)), str(np.std(ls)), str(np.mean(vels_1)), 
                             str(np.std(vels_1)), str(np.mean(vels_2)), str(np.std(vels_2))])
