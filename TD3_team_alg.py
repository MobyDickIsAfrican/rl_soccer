from spinup.algos.pytorch.td3.td3 import td3_soccer_game
from utils.stage_wrappers_env import stage_soccerTraining 
from numpy.lib.npyio import save
import dm_soccer2gym.wrapper as wrap
from torch import nn
import torch
from torch.optim import Adam
import numpy as np
from copy import deepcopy
import itertools
import time
from spinup.utils.logx import EpochLogger

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.



class MLPAC_4_team(nn.Module):

    def __init__(self, team, players, observation_space, action_space, loss_dict, polyak=0.1,  hidden_sizes=(256, 256),
                activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.polyak = polyak
        self.loss_dict = loss_dict
        self.team = team

        # build policy for each player in team
        self.pi = nn.ModuleList([MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)\
                                                         for _ in range(players)])
        
        # build critic: 
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, critic_using=False):
        with torch.no_grad():
            if critic_using:
                return torch.cat([self.pi[i](obs[i])[np.newaxis, :] for i in range(len(self.pi))], axis=0)
            else:
                return torch.cat([self.pi[i](obs)[np.newaxis, :] for i in range(len(self.pi))], axis=0)

    def compute_q_loss(self, data, ac_targ):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.q1(o,a)
        q2 = self.q2(o,a)

        # get training params: 
        target_noise = self.loss_dict["target_noise"]
        noise_clip = self.loss_dict["noise_clip"]
        act_limit = self.loss_dict["act_limit"]
        gamma = self.loss_dict["gamma"]


        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.act(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = torch.cat([ac_targ.q1(o2, a2[i])[np.newaxis, :] for i in range(a2.shape[0])])
            q2_pi_targ = torch.cat([ac_targ.q2(o2, a2[i])[np.newaxis, :] for i in range(a2.shape[0])])
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, data):
        o = [data_i['obs'] for data_i in data]
        q1_pi = [self.q1(o[i], self.pi[i](o[i])) for i in range(len(self.pi))]
        return torch.sum(torch.cat([-q1_i.mean()[np.newaxis] for q1_i in q1_pi]))

    # update method for the networks: 
    def update(self, team_buffer, critic_buffer, q_optim, pi_optim, ac_targ, timer, logger, q_param, policy_delay):
        q_optim.zero_grad()
        loss_q, loss_info = self.compute_q_loss(critic_buffer, ac_targ)
        loss_q.backward()
        q_optim.step()
        # Record things
        logger.store(team=self.team, LossQ=loss_q.item(),**loss_info)
        if timer % policy_delay:
            for p in q_param:
                p.requires_grad = False

            pi_optim.zero_grad()
            loss_pi = self.compute_loss_pi(team_buffer)
            loss_pi.backward()
            pi_optim.step()

            # unfreeze critics: 
            for p in q_param:
                p.requieres_grad = True
            
            # Record things
            logger.store(team=self.team, LossPi=loss_pi.item()) 

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.pi.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

class TD3_team_alg:
    def __init__(self, env_fn, home_players, away_players, actor_critic=MLPAC_4_team, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=4e-3, q_lr=4e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1) -> None: 

        self.home = home_players
        self.away = away_players
        self.__name__ = "training"
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.replay_size = replay_size
        self.act_noise = act_noise
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]
        self.loss_param_dict = {'target_noise': target_noise,
                                'noise_clip': noise_clip,
                                'act_limit': act_limit, 
                                'gamma': gamma}
        
        self.training_param_dict = {"epochs": epochs,
                                    "steps_per_epoch": steps_per_epoch,
                                    "polyak": polyak,
                                    "batch_size": batch_size,
                                    "start_steps": start_steps, 
                                    "update_after": update_after, 
                                    "update_every": update_every,
                                    "num_test_episodes": num_test_episodes,
                                    "max_ep_len": max_ep_len,
                                    "policy_delay": policy_delay,
                                    "save_freq": save_freq
                                    }


        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        

        #### CREATION OF HOME TEAM ##########################
        # Create actor-critic module and target networks for each team:
        # create actor critic agent for home team
        self.home_ac, self.home_ac_targ, self.home_q_params, self.home_team_buffer, self.home_critic_buffer\
                    , self.home_var_counts = self.create_team("home",home_players, actor_critic, ac_kwargs)  

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = list(count_vars(module) for module in [*self.home_ac.pi, self.home_ac.q1, self.home_ac.q2])
        self.logger.log(f'\nNumber of parameters for home team: \t pi: {var_counts[:-2]}, \t q1: {var_counts[-2]}, \t q2: {var_counts[-1]}\n')

        ##### CREATION OF AWAY TEAM #######################
        self.away_ac, self.away_ac_targ, self.away_q_params, self.away_team_buffer, self.away_critic_buffer\
                     , self.away_var_counts= self.create_team("away", away_players, actor_critic, ac_kwargs) 

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = list(count_vars(module) for module in [*self.away_ac.pi, self.away_ac.q1, self.away_ac.q2])
        self.logger.log(f'\nNumber of parameters for away team: \t pi: {var_counts[:-2]}, \t q1: {var_counts[-2]}, \t q2: {var_counts[-1]}\n')

        # Set up optimizers for policy and q-function for the home team
        self.home_pi_optimizer = Adam(self.home_ac.pi.parameters(), lr=pi_lr)
        self.home_q_optimizer = Adam(self.home_q_params, lr=q_lr)

        # Set up optimizers for policy and q-function for the away team:
        self.away_pi_optimizer = Adam(self.away_ac.pi.parameters(), lr=pi_lr)
        self.away_q_optimizer = Adam(self.away_q_params, lr=q_lr)
        
        

    def create_team(self, home_or_away, n_players, actor_critic, ac_kwargs):
        # Create actor-critic module and target networks for each team:
        # create actor critic agent for home team
        polyak = self.training_param_dict['polyak']
        ac = actor_critic(home_or_away, n_players, self.env.observation_space, self.env.action_space, self.loss_param_dict, polyak, **ac_kwargs)
        ac_targ = deepcopy(ac)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False
        
        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

        ## creation of replay buffers, we need n_players replay buffers + 1 replay buffer for the critics:
        # Experience buffer:
        team_buffer = [ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)\
                            for _ in range(n_players)]
        critic_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        var_counts = list(count_vars(module) for module in [*ac.pi, ac.q1, ac.q2])
        return ac, ac_targ, q_params, team_buffer, critic_buffer, var_counts
    
    def compute_q_loss(self, q_home_data, q_away_data):
        return [self.home_ac.compute_q_loss(q_home_data, self.home_ac_targ, self.loss_param_dict),\
                self.away_ac.compute_q_loss(q_away_data, self.away_ac_targ, self.loss_param_dict)]

    def compute_loss_pi(self, data_home, data_away):
        return [self.home_ac.compute_loss_pi(data_home),
                self.away_ac.compute_loss_pi(data_away)]
    
    def update(self, data_home_team, data_home_q, data_away_team, data_away_q, timer):
        polyak = self.training_param_dict['polyak']
        self.home_ac.update(data_home_team, data_home_q,self.home_q_optimizer, self.home_pi_optimizer, self.home_ac_targ,\
                            timer, self.logger,self.home_q_params, polyak)

        self.away_ac.update(data_away_team, data_away_q, self.away_q_optimizer, self.away_pi_optimizer, self.away_ac_targ,\
                            timer, self.logger, self.away_q_params, polyak)


    def get_action(self, o, noise_scale):
        act_lim = self.loss_param_dict['act_limit']
        critic_using = not (o.shape[0] == self.training_param_dict["batch_size"])
        actions = torch.cat([self.home_ac.act(torch.as_tensor(o[:self.home], dtype=torch.float32), critic_using),\
                  self.away_ac.act(torch.as_tensor(o[self.home:], dtype=torch.float32), critic_using)]).numpy()
        actions += noise_scale*np.random.randn(self.act_dim)
        return np.clip(actions, -act_lim, act_lim)
    
    def test_agent(self):
        num_test_episodes = self.training_param_dict["num_test_episodes"]
        max_ep_len = self.training_param_dict["max_ep_len"]
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, np.array([0]*(self.home + self.away), dtype='float32'), 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


    def train_agents(self):
        
        epochs = self.training_param_dict["epochs"]
        steps_per_epoch = self.training_param_dict["steps_per_epoch"]
        save_freq = self.training_param_dict["save_freq"]
        total_steps = steps_per_epoch*epochs
        start_steps = self.training_param_dict["start_steps"]
        start_time = time.time()
        max_ep_len = self.training_param_dict["max_ep_len"]
        o, ep_ret, ep_len = self.env.reset(), np.array([0]*(self.home + self.away), dtype='float32'), 0

        for t in range(total_steps):
            print(t)
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise). 
            if t > start_steps:
                a = self.get_action(o, self.act_noise)
            else:
                a = [self.env.action_space.sample() for _ in range(self.home+self.away)]

            # step in the env:
            o2, r, d, _ = self.env.step(a)
            ep_ret += np.array(r)
            ep_len += 1

            d = False if ep_len == max_ep_len else d

            if d or (ep_len == max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), np.array([0]*(self.home + self.away), dtype='float32'), 0

            # store in buffer: 
            for i, home_buf in enumerate(self.home_team_buffer):
                home_buf.store(o[i], a[i], r[i], o2[i], d)
                self.home_critic_buffer.store(o[i], a[i], r[i], o2[i], d)
            off_set = self.home
            for j, away_buf in enumerate(self.away_team_buffer):
                i = j+ off_set
                away_buf.store(o[i], a[i], r[i], o2[i], d)
                self.away_critic_buffer.store(o[i], a[i], r[i], o2[i], d)

            # Update handling
            if t >= self.training_param_dict[ "update_after"] and t % self.training_param_dict["update_every"] == 0:
                for j in range(self.training_param_dict["update_every"]):
                    # sample from home buffers: 
                    batch_home = [buf.sample_batch(self.training_param_dict["batch_size"])\
                                  for buf in self.home_team_buffer]
                    batch_q_home = self.home_critic_buffer.sample_batch(self.training_param_dict["batch_size"])

                    # sample from away buffers: 
                    batch_away = [buf.sample_batch(self.training_param_dict["batch_size"])\
                                  for buf in self.away_team_buffer]

                    batch_q_away = self.away_critic_buffer.sample_batch(self.training_param_dict["batch_size"])

                    self.update(batch_home, batch_q_home, batch_away, batch_q_away, timer=j)

            if (t+1)% steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch

                if (epoch% save_freq ==0) or (epoch==epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()
                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()

                
class soccer2vs0(TD3_team_alg):
    def __init__(self, env_fn, home_players, actor_critic=MLPAC_4_team, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=10000, epochs=2000, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=4e-3, q_lr=4e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1) -> None:

        
        self.home = home_players
        self.__name__ = "training"
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.replay_size = replay_size
        self.act_noise = act_noise
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]
        self.loss_param_dict = {'target_noise': target_noise,
                                'noise_clip': noise_clip,
                                'act_limit': act_limit, 
                                'gamma': gamma}
        
        self.training_param_dict = {"epochs": epochs,
                                    "steps_per_epoch": steps_per_epoch,
                                    "polyak": polyak,
                                    "batch_size": batch_size,
                                    "start_steps": start_steps, 
                                    "update_after": update_after, 
                                    "update_every": update_every,
                                    "num_test_episodes": num_test_episodes,
                                    "max_ep_len": max_ep_len,
                                    "policy_delay": policy_delay,
                                    "save_freq": save_freq
                                    }


        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        

        #### CREATION OF HOME TEAM ##########################
        # Create actor-critic module and target networks for each team:
        # create actor critic agent for home team
        self.home_ac, self.home_ac_targ, self.home_q_params, self.home_team_buffer, self.home_critic_buffer\
                    , self.home_var_counts = self.create_team("home",home_players, actor_critic, ac_kwargs)  

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = list(count_vars(module) for module in [*self.home_ac.pi, self.home_ac.q1, self.home_ac.q2])
        self.logger.log(f'\nNumber of parameters for home team: \t pi: {var_counts[:-2]}, \t q1: {var_counts[-2]}, \t q2: {var_counts[-1]}\n')

        # Set up optimizers for policy and q-function for the home team
        self.home_pi_optimizer = Adam(self.home_ac.pi.parameters(), lr=pi_lr)
        self.home_q_optimizer = Adam(self.home_q_params, lr=q_lr)


    def compute_q_loss(self, q_home_data):
        return self.home_ac.compute_q_loss(q_home_data, self.home_ac_targ, self.loss_param_dict)

    def compute_loss_pi(self, data_home):
        return self.home_ac.compute_loss_pi(data_home)
    
    def update(self, data_home_team, data_home_q, timer):
        polyak = self.training_param_dict['polyak']
        self.home_ac.update(data_home_team, data_home_q,self.home_q_optimizer, self.home_pi_optimizer, self.home_ac_targ,\
                            timer, self.logger,self.home_q_params, polyak)


    def get_action(self, o, noise_scale):
        act_lim = self.loss_param_dict['act_limit']
        critic_using = not (o.shape[0] == self.training_param_dict["batch_size"])
        actions = self.home_ac.act(torch.as_tensor(o[:self.home], dtype=torch.float32), critic_using).numpy()
        actions += noise_scale*np.random.randn(self.act_dim)
        return np.clip(actions, -act_lim, act_lim)


    def test_agent(self):
        num_test_episodes = self.training_param_dict["num_test_episodes"]
        max_ep_len = self.training_param_dict["max_ep_len"]
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, np.array([0]*(self.home), dtype='float32'), 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agents(self):
        
        epochs = self.training_param_dict["epochs"]
        steps_per_epoch = self.training_param_dict["steps_per_epoch"]
        save_freq = self.training_param_dict["save_freq"]
        total_steps = steps_per_epoch*epochs
        start_steps = self.training_param_dict["start_steps"]
        start_time = time.time()
        max_ep_len = self.training_param_dict["max_ep_len"]
        o, ep_ret, ep_len = self.env.reset(), np.array([0]*(self.home), dtype='float32'), 0

        for t in range(total_steps):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise). 
            if t > start_steps:
                a = self.get_action(o, self.act_noise)
            else:
                a = [self.env.action_space.sample() for _ in range(self.home)]

            # step in the env:
            o2, r, d, _ = self.env.step(a)
            ep_ret += np.array(r)
            ep_len += 1

            d = False if ep_len == max_ep_len else d

            if d or (ep_len == max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), np.array([0]*(self.home), dtype='float32'), 0

            # store in buffer: 
            for i, home_buf in enumerate(self.home_team_buffer):
                home_buf.store(o[i], a[i], r[i], o2[i], d)
                self.home_critic_buffer.store(o[i], a[i], r[i], o2[i], d)

            # Update handling
            if t >= self.training_param_dict[ "update_after"] and t % self.training_param_dict["update_every"] == 0:
                for j in range(self.training_param_dict["update_every"]):
                    # sample from home buffers: 
                    batch_home = [buf.sample_batch(self.training_param_dict["batch_size"])\
                                  for buf in self.home_team_buffer]
                    batch_q_home = self.home_critic_buffer.sample_batch(self.training_param_dict["batch_size"])

                    self.update(batch_home, batch_q_home, timer=j)

            if (t+1)% steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch

                if (epoch% save_freq ==0) or (epoch==epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()
                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='td3_soccer')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, "result_soccer")
    env_creator = lambda :   stage_soccerTraining(team_1=2, team_2=0) 
    T3 = soccer2vs0(env_creator, 2, logger_kwargs= logger_kwargs)   
    T3.train_agents()      



    
