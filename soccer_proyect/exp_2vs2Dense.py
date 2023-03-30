import argparse
from email.mime import base
from spinup.utils.logx import EpochLogger
from math import ceil
from learning_algorithm.TD3_team_alg_concat import TD3_team_alg
from environments.env2vs2Dense import Env2vs2
import json
import random
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--model_path', type=str, default="/models")
parser.add_argument('--save_path', type=str, default="/results/2vs2Train")
parser.add_argument('--exp_name', type=str, default='td3_soccer')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument("--control_timestep", type=float, default=0.1)
parser.add_argument("--time_limit", type=float, default=30.)
args = parser.parse_args()
args = parser.parse_args()


from spinup.utils.run_utils import setup_logger_kwargs
#model_path = "//home//amtc//roberto//Proyecto//1vs0//2022-05-02_11-58-33_td3_soccer_goal_pass_concat_2vs0_0.1//pyt_save//model2999999.pt"
base_path = args.model_path
base_path = "D:\\rl_soccer\\2vs0\\Junio_test_2022"
with open(os.path.join(base_path, "selected_models.json"), encoding='utf-8') as json_file:
        agents_json = json.load(json_file).values()
        agents = []
        for a_run in agents_json:
            agents+=a_run

rivals = [agents[22][0].split("\\"), agents[21][0].split("\\"), agents[19][0].split("\\")]
rivals = [os.path.join(base_path, *a_rival) for a_rival in rivals]
exp_kwargs = {"free_play":True}
logger_kwargs = setup_logger_kwargs(f"td3_soccer_goal_orig_concat_2vs0_{args.control_timestep}", data_dir=args.save_path, datestamp=True)
env_creator = lambda :   Env2vs2(team_1=2, team_2=2,task_kwargs={ "time_limit": args.time_limit, "disable_jump": True, 
    "dist_thresh": 0.03, 'control_timestep': args.control_timestep,  "observables": "all"}) 
env_test_creator = lambda : Env2vs2(team_1=2, team_2=2, task_kwargs={"time_limit": args.time_limit, "disable_jump": True, 
    "dist_thresh": 0.03, 'control_timestep': 0.1, 'random_state': 69,  "observables": "all"})
T3 = TD3_team_alg(2, 2,env_creator,  epochs=300,logger_kwargs= logger_kwargs, max_ep_len=ceil(args.time_limit / args.control_timestep), 
    test_fn=env_test_creator,  exp_kwargs=exp_kwargs)   
T3.train_agents()     