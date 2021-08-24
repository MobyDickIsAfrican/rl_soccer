import argparse
from spinup.utils.logx import EpochLogger
from math import ceil
from TD3_team_alg import soccer2vs0
from utils.stage_wrappers_env import stage_soccerTraining
parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--exp_name', type=str, default='td3_soccer')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument("--control_timestep", type=float, default=0.1)
parser.add_argument("--time_limit", type=float, default=30.)
args = parser.parse_args()
args = parser.parse_args()

from spinup.utils.run_utils import setup_logger_kwargs
logger_kwargs = setup_logger_kwargs(f"td3_soccer_goal_orig_join_2vs0_{args.control_timestep}", data_dir="roberto/Proyecto/2vs0", datestamp=True)
env_creator = lambda :   stage_soccerTraining(team_1=2, team_2=0,task_kwargs={ "time_limit": args.time_limit, "disable_jump": True, 
    "dist_thresh": 0.03, 'control_timestep': args.control_timestep,  "observables": "all"}) 
env_test_creator = lambda : stage_soccerTraining(team_1=2, team_2=0, task_kwargs={"time_limit": args.time_limit, "disable_jump": True, 
    "dist_thresh": 0.03, 'control_timestep': 0.1, 'random_state': 69, "observables": "all"})
T3 = soccer2vs0(env_creator, 2, epochs=300,logger_kwargs= logger_kwargs, max_ep_len=ceil(args.time_limit / args.control_timestep), test_fn=env_test_creator)   
T3.train_agents()     