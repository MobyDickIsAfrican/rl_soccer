
import argparse
import dm_soccer2gym
import numpy as np
import os
from spinup.utils.test_policy import load_policy_and_env
import tensorflow as tf


num_runs = 125
max_ep_len = 600
RENDER = False


def main(agent_1, agent_2, sess=None, order='home_away'):

    env = dm_soccer2gym.make("2vs2goal", task_kwargs={'rew_type': 'simple_v2', 'time_limit': 45., 'disable_jump': True, 'dist_thresh': .03, 
                             'control_timestep': 0.05, 'random_state': 42})
    
    if order == 'home_away':
        
        _, get_action_1 = load_policy_and_env(*agent_1, sess=sess)
            
        g2 = tf.Graph()
        
        with g2.as_default():
            _, get_action_2 = load_policy_and_env(*agent_2, sess=None)
        
        g3 = tf.Graph()
        
        with g3.as_default():
            _, get_action_3 = load_policy_and_env(*agent_3, sess=None)
        
        g4 = tf.Graph()
        
        with g2.as_default():
            _, get_action_4 = load_policy_and_env(*agent_4, sess=None)
    
    elif order == 'away_home':

        _, get_action_3 = load_policy_and_env(*agent_1, sess=sess)
            
        g2 = tf.Graph()
        
        with g2.as_default():
            _, get_action_4 = load_policy_and_env(*agent_2, sess=None)
        
        g3 = tf.Graph()
        
        with g3.as_default():
            _, get_action_1 = load_policy_and_env(*agent_3, sess=None)
        
        g4 = tf.Graph()
        
        with g2.as_default():
            _, get_action_2 = load_policy_and_env(*agent_4, sess=None)
    
    else:
        raise ValueError("Unexpected value for  parameter order, received %s, expected home_away | away_home" % (order))

    s = 0
    f = 0
    t = 0

    for j in range(num_runs):
            
        obs = env.reset()
        d = False
        l = 0

        while not(d or (l == max_ep_len)):
            a = [get_action_1(obs[0][np.r_[0:18, 18:24]]), get_action_2(obs[1][np.r_[0:18, 24:30]]),
                 get_action_3(obs[2][np.r_[0:18, 18:24]]), get_action_4(obs[3][np.r_[0:18, 24:30]])]
            if RENDER:
                env.render()
            obs, r, d, _ = env.step(a)
            l += 1

        if env.timestep.reward[0] > 0: s += 1
        elif env.timestep.reward[0] < 0: f += 1
        else: t += 1
            
        if (j + 1) % 10 == 0: 
            print(j + 1, s, t, f)

    return s, t, f


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("agents_path", type=str, help="File where agents model_path and num are")
    parser.add_argument("--model_path", type=str, default=".", help="Path where models are")
    parser.add_argument("--agent_1_idx", type=int, help="Agent 1 (team 1) index")
    parser.add_argument("--agent_2_idx", type=int, help="Agent 2 (team 1) index")
    parser.add_argument("--agent_3_idx", type=int, help="Agent 3 (team 1) index")
    parser.add_argument("--agent_4_idx", type=int, help="Agent 4 (team 1) index")
    parser.add_argument("--gpu", type=float, help="Fraction of gpu to use", default=1.0)
    parser.add_argument("--save_path", type=str, help="Path to where save file with results", 
                        default=".")
    parser.add_argument("--order", type=str, help="Order to be used for agents, home_away | away_home",
                        default='home_away')
    
    args = parser.parse_args()

    agents = getattr(__import__(args.agents_path), 'AGENTS')
    
    orig_path = os.getcwd()
    os.chdir(args.model_path)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    s, t, f = main(agents[args.agent_1_idx], agents[args.agent_2_idx], 
                   agents[args.agent_3_idx], agents[args.agent_4_idx], sess, args.order)
    
    os.chdir(orig_path)
    os.makedirs(args.save_path, exist_ok=True)
    filename = f'{min(args.agent_1_idx, args.agent_2_idx)}_{max(args.agent_1_idx, args.agent_2_idx)}' +\
        f'vs{min(args.agent_3_idx, args.agent_4_idx)}_{max(args.agent_3_idx, args.agent_4_idx)}.txt'
        
    with open(os.path.join(args.save_path, filename), 'a') as f_:
        if args.order == 'home_away':
            f_.write("%d %d %d\n" % (s, t, f))
        else:
            f_.write("%d %d %d\n" % (f, t, s))

