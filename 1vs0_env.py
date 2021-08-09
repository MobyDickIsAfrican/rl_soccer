import dm_soccer2gym
from TD3_team_alg import soccer2vs0
import argparse
from math import ceil


parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--exp_name', type=str, default='td3_soccer')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument("--reward", type=str, default="simple_v2")
parser.add_argument("--control_timestep", type=float, default=0.1)
parser.add_argument("--time_limit", type=float, default=30.)
args = parser.parse_args()
args = parser.parse_args()

env_creator = lambda : dm_soccer2gym.make('1vs0goal', task_kwargs={"rew_type": args.reward, "time_limit": args.time_limit, "disable_jump": True, 
                "dist_thresh": 0.03, 'control_timestep': 0.1})
env_test_creator = lambda : dm_soccer2gym.make('1vs0goal', task_kwargs={"rew_type": "simple_v2", "time_limit": args.time_limit, "disable_jump": True, 
        "dist_thresh": 0.03, 'control_timestep': 0.1, 'random_state': 69})

from spinup.utils.run_utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, "result_soccer")
T3 = soccer2vs0(env_creator,  1, logger_kwargs= logger_kwargs, epochs= 200, max_ep_len=ceil(args.time_limit / 0.1), test_fn=env_test_creator)   
T3.train_agents()  