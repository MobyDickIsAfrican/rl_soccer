import argparse
from spinup.utils.logx import EpochLogger
from math import ceil
from learning_algorithm.TD3_team_alg_concat import TD3_team_alg
from environments.env2vs0 import Env2vs0 
parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--save_path', type=str, default="/results/2vs2Train")
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument("--control_timestep", type=float, default=0.1)
parser.add_argument("--time_limit", type=float, default=30.)
args = parser.parse_args()
args = parser.parse_args()


from spinup.utils.run_utils import setup_logger_kwargs

model_path = "trainedBaseModels/model1vs0_0.pt"
rivals = [model_path]
exp_kwargs = {"free_play":False, "rivals": None,  "actor_state_dict": rivals[0],'train_now': ["teammateEnc"]}
logger_kwargs = setup_logger_kwargs(f"td3_2vs0", data_dir=args.save_path, datestamp=True)
env_creator = lambda :   Env2vs0(team_1=2, team_2=0,task_kwargs={ "time_limit": args.time_limit, "disable_jump": True, 
    "dist_thresh": 0.03, 'control_timestep': args.control_timestep,  "observables": "all"}) 
env_test_creator = lambda : Env2vs0(team_1=2, team_2=0, task_kwargs={"time_limit": args.time_limit, "disable_jump": True, 
    "dist_thresh": 0.03, 'control_timestep': 0.1, 'random_state': 69,  "observables": "all"})

T3 =TD3_team_alg(2, 0, env_creator, epochs=300,logger_kwargs= logger_kwargs, max_ep_len=ceil(args.time_limit / args.control_timestep), 
    test_fn=env_test_creator,  exp_kwargs=exp_kwargs)   
T3.train_agents()     

