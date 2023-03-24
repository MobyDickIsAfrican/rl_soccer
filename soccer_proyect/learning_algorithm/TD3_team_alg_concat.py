from dm_control.composer.define import observable
from torch._C import device
from torch.nn.modules.activation import LeakyReLU
from spinup.algos.pytorch.td3.td3 import td3_soccer_game
from numpy.lib.npyio import save
import dm_soccer2gym.wrapper as wrap
from torch import nn
import torch
from torch.optim import Adam
from spinup.utils.test_policy import load_policy_and_env
import numpy as np
from copy import deepcopy
import itertools
import time
from spinup.utils.logx import EpochLogger
from tqdm.auto import tqdm
import random

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size, rew_dim):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size,  obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(combined_shape(size, rew_dim), dtype=np.float32)
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



def mlp(sizes, activation, output_activation=nn.Identity, init=None):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        linear = nn.Linear(sizes[j], sizes[j+1])
        if init:
            torch.nn.init.normal_(linear.weight, mean=0, std=1e-4)
        layers += [linear, act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, teammates, rivals):
        super().__init__()
        # setting the amount of players that will play
        self.teammates = teammates
        self.rivals = rivals
        # setting the obs_analyzer: its 9 mlp per actor that process a concatenation of obs_i
        # the first list is for propioceptive observations and the second list is for external players
        outputEncDim = 9*64
        # setting the size of the policy network:
        # where obs_dim is the dimension of the observations after going through the mlp
        # hidden_sizes is a list that shows the amount of hidden layers
        # act_dim is the action dimension of the players.
        pi_sizes = [outputEncDim] + list(hidden_sizes) + [act_dim]
        self.rival_offset = 18+12*self.teammates
        # setting the mlp
        self.pi = mlp(pi_sizes, activation, output_activation=nn.Tanh)
        # propiocentric observation:
        propEnc =[mlp([2, 32, 64], activation=nn.LeakyReLU, output_activation=nn.LeakyReLU) for _ in range(9)] 
        # teammate observation:
        teammateEnc = [mlp([12, 32, outputEncDim], activation=nn.LeakyReLU, output_activation=nn.LeakyReLU, init=True) for _ in range(teammates)]
        # rivalObservation
        rivalEnc = [mlp([9, 32, outputEncDim], activation=nn.LeakyReLU, output_activation=nn.LeakyReLU, init=True) for _ in range(rivals)]

        self.propEncoder = nn.ModuleList(propEnc)
        self.teammateEncoder = nn.ModuleList(teammateEnc)
        self.rivalEncoder = nn.ModuleList(rivalEnc)
        self.act_limit = torch.Tensor(np.array(act_limit)).cpu()
    
    def encode(self, obs):
        """ 
        encode is a function that generates the encoding of the observations into a feature space that 
        is then used by the decoder part of the network. 
        Args:
            obs (torch.Tensor): The mujoco observations that are generated for every agent
            act (torch.Tensor): The actions taken by the agent in time t-1. 

        Returns:
            torch.Tensor: The encoded observations
        """
        # prop observations
        obs = torch.cat([self.propEncoder[i](obs[:, 2*i:2*(i+1)]) for i in range(9)], -1)
        if self.teammates>0:
            # teammate observation
            obs += torch.cat([self.teammateEncoder[i](obs[:, 18 + 12*i: 18 + 12*(i+1)]) for i in range(self.teammates)], -1)
        if self.rivals>0:
            # rival observation
            obs += torch.cat([self.rivalEncoder[i](obs[obs[:, self.rival_offset+ 9*i: self.rival_offset + 9*(i+1)]]) for i in range(self.rivals)], -1)
        return obs

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        encoded_obs = self.encode(obs)
        return self.act_limit * self.pi(encoded_obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, action_dim,hidden_sizes, activation, teammates, rivals):
        super().__init__()
        # amount of players in the team
        self.n_players = teammates+1
        # number of teammates
        self.teammates = teammates
        # number of rivals
        self.rival = rivals
        # OBSERVARTIONS.
        # OFFSETS:
        self.prop_offset = 2
        self.teammate_offset = 12
        self.rival_offset = 9
        # START OF OBSERVATION
        self.rival_start = 9*self.rival_offset + 12*self.n_players

        #MODEL GENERATION
        # DECODER
        self.q = mlp([self.n_players*obs_dim] + list(hidden_sizes) + [self.n_players], activation)
        # PROPIOCENTRIC ENCODER
        propEnc = [mlp([self.prop_offset+action_dim,32, 64], activation=nn.LeakyReLU, output_activation=nn.LeakyReLU) for _ in range(9)]
        # TEAMMATE OBSERVATION ENCODER
        teammateEnc = [mlp([12, 32, 64], activation=nn.LeakyReLU, output_activation=nn.LeakyReLU) for _ in range(teammates)]
        #RIVAL ENCODER
        rivalEnc =  [mlp([9, 32, 64], activation=nn.LeakyReLU, output_activation=nn.LeakyReLU) for _ in range(rivals)]
        # FINAL MODEL
        self.obs_analyzer = nn.ModuleList(propEnc+teammateEnc+rivalEnc)



    def encode(self, obs, act):
        """ 
        encode is a function that generates the encoding of the observations into a feature space that 
        is then used by the decoder part of the network. 
        Args:
            obs (torch.Tensor): The mujoco observations that are generated for every agent
            act (torch.Tensor): The actions taken by the agent in time t-1. 

        Returns:
            torch.Tensor: The encoded observations
        """
        encoded_obs = []
        # iterate through every player
        for player in range(self.n_players): 
            # generate the propioceptive encoded observations of that player
            obs_prop = [self.obs_analyzer[i](torch.cat([obs[:, player, self.prop_offset*i : self.prop_offset*(i+1)], act[:, player, :]], 1)) for i in range(9)]
            # generate the teammate encoded observations of that player
            obs_teammate = [self.obs_analyzer[9 + i](obs[:, player, self.prop_offset*9 + self.teammate_offset*i : self.prop_offset*9 + self.teammate_offset*(i+1)])\
                                 for i in range(self.teammates)]
            # generate the rival encoded observations:
            obs_rival = [self.obs_analyzer[(9+self.teammates) + i](obs[:, player, self.rival_start+ self.rival_offset*i : self.rival_start + self.rival_offset*(i+1)])\
                                 for i in range(self.rival)]
            # concatenate all observations of player
            encoded_obs.append(torch.cat(obs_prop + obs_teammate + obs_rival, -1))
        # concatenate all observations
        return torch.cat(encoded_obs, -1)


    def forward(self, obs, act):
        encoded_obs = self.encode(obs, act)
        q = self.q(encoded_obs)
        return q # Critical to ensure q has right shape.



class MLPAC_4_team(nn.Module):
    def calculate_obs(self, teammates, rivals):
        '''
        prop_dims = 18
        teammate_dims = teammates*12
        opponent_dims = rivals*9
            
        '''
        return (9+ teammates+ rivals)*64

        

    def __init__(self, team, home, away, observation_space, action_space, loss_dict, polyak=0.1,  hidden_sizes=(256, 256),
                activation=nn.LeakyReLU):
        super().__init__()
        if team =="home":
            teammates = home-1
            rivals = away
        else:
            teammates = away-1
            rivals = home
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.polyak = polyak
        self.loss_dict = loss_dict
        self.team = team
        self.total_delay = 0
        self.actual_delay = 0
        players = teammates+1

        obs_dim= self.calculate_obs(teammates, rivals)
        # build policy for each player in team
        self.pi = nn.ModuleList([MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, teammates, rivals)\
                                                         for _ in range(players)])
        
        # build critic: 
        # 9 for the proprioceptive measurements of each agent, players-1 for each teammate measurement and 
        # one for the rivals.
        critic_action_dim = act_dim
        
        #obs_dim, hidden_sizes, activation, teammates, rivals
        self.q1 = MLPQFunction(obs_dim, critic_action_dim,hidden_sizes, activation, teammates, rivals)
        self.q2 = MLPQFunction(obs_dim, critic_action_dim,hidden_sizes, activation, teammates, rivals)

    def act(self, obs):
        return torch.cat([torch.unsqueeze(self.pi[i](obs[:, i, :]),1) for i in range(len(self.pi))], axis=1)

    def compute_q_loss(self, data, ac_targ):
        o, a, r, o2, d = torch.Tensor(data['obs']).cuda(), torch.Tensor(data['act']).cuda(), torch.Tensor(np.array(data['rew'])).cuda(),\
                         torch.Tensor(data['obs2']).cuda(), torch.Tensor(data['done']).cuda()
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
            epsilon = torch.normal(0, target_noise, size=pi_targ.shape, device=pi_targ.device)
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.minimum(q1_pi_targ, q2_pi_targ)
            backup = r + (gamma * (1 - d[..., np.newaxis]) * q_pi_targ)

        # MSE loss against Bellman backup
        loss_q1 = torch.sum(torch.mean(torch.square((q1 - backup)), 0))
        loss_q2 = torch.sum(torch.mean(torch.square((q2 - backup)), 0))
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy())

        return loss_q, loss_info

    

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, data):
        # get the observation vector:
        o = torch.Tensor(data["obs"]).cuda()
        # compute the Q value of the state, action: 
        q1_pi = self.q1(o, self.act(o))
        # get the mean value of the q1_pi values and turn them negative
        # for gradient descent
        return -q1_pi.mean()

    # update method for the networks: 
    def update(self, buffer, q_optim, pi_optim, ac_targ, timer, logger, q_param, policy_delay):
        q_optim.zero_grad()
        loss_q, loss_info = self.compute_q_loss(buffer, ac_targ)
        loss_q.backward()
        q_optim.step()
        # Record things
        logger.store(team=self.team, LossQ=loss_q.item(), **loss_info)
        if (timer % policy_delay) ==0:

            # set gradiente to zero:
            if self.actual_delay >= 2*self.total_delay:
                # freeze critic: 
                self.q1.eval()

                pi_optim.zero_grad()
                # calculate loss: 
                loss_pi = self.compute_loss_pi(buffer)

                #calculate backward grads
                loss_pi.backward()
                # update weights:
                pi_optim.step()

                # unfreeze critics: 
                self.q1.train()

                # Record things
                logger.store(team=self.team, LossPi=loss_pi.item()) 
            else:
                logger.store(team=self.team, LossPi=0) 

            
            

            
class TD3_team_alg:
    def __init__(self, home, away,env_fn, actor_critic=MLPAC_4_team, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=10000, epochs=2000, replay_size=int(2e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=256, start_steps=50000, 
        update_after=10000, update_every=50, act_noise=0.1, target_noise=0.1, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=50, max_ep_len=300, 
        logger_kwargs=dict(), save_freq=10, test_fn=None, exp_kwargs=dict()) -> None: 

        self.home = home
        self.away = away
        self.__name__ = "training"
        self.env, self.test_env = env_fn(), test_fn() if test_fn is not None else env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.replay_size = replay_size
        self.act_noise = act_noise

        ## is it freeplay?
        self.free_play = exp_kwargs.get("free_play", False)
        actor_state_dict = exp_kwargs.get("actor_state_dict", None)
        if not self.free_play:
            selected_rivals = exp_kwargs.get("rivals", None)


        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]
        # generate a parameter dict in which hyperparameters are stored
        self.loss_param_dict = {'target_noise': target_noise,
                                'noise_clip': noise_clip,
                                'act_limit': act_limit, 
                                'gamma': gamma}
        # generate another dict where some specifics of training are stored
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
        self.home_ac, self.home_ac_targ, self.home_q_params, self.home_critic_buffer\
                    , self.home_var_counts= self.create_team("home", self.home, actor_critic, ac_kwargs, actor_state_dict=actor_state_dict)  

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = list(count_vars(module) for module in [*self.home_ac.pi, self.home_ac.q1, self.home_ac.q2])
        self.logger.log(f'\nNumber of parameters for home team: \t pi: {var_counts[:-2]}, \t q1: {var_counts[-2]}, \t q2: {var_counts[-1]}\n')

        ##### CREATION OF AWAY TEAM #######################
        if away>0:
            if self.free_play:
                self.away_ac, self.away_ac_targ, self.away_q_params, self.away_critic_buffer\
                            , self.away_var_counts= self.create_team("away", self.away, actor_critic, ac_kwargs, actor_state_dict=actor_state_dict) 

                # Count variables (protip: try to get a feel for how different size networks behave!)
                var_counts = list(count_vars(module) for module in [*self.away_ac.pi, self.away_ac.q1, self.away_ac.q2])
                self.logger.log(f'\nNumber of parameters for away team: \t pi: {var_counts[:-2]}, \t q1: {var_counts[-2]}, \t q2: {var_counts[-1]}\n')
                # Set up optimizers for policy and q-function for the away team:
                self.away_pi_optimizer = Adam(self.away_ac.pi.parameters(), lr=pi_lr)
                self.away_q_optimizer = Adam(self.away_q_params, lr=q_lr)
            else:
                self.rivals = [load_policy_and_env(a_rival)[1] for a_rival in selected_rivals]
                self.away_ac = self.rivals.pop(0)

        # Set up optimizers for policy and q-function for the home team
        if not (actor_state_dict is None) and (self.away==0 or not self.free_play):
            pi_parameters = list(self.home_ac.pi.named_parameters())
            pi_trained_params, pi_train_now_params = list(), list()
            for name, parameter in pi_parameters:
                if any(map(lambda x: x in name, exp_kwargs.get("train_now", []))):
                    pi_train_now_params.append(parameter)
                else: 
                    pi_trained_params.append(parameter)
            self.home_pi_optimizer = Adam([{'params': pi_train_now_params}, {'params': pi_trained_params, 'lr':pi_lr*1e-1}], lr=pi_lr)
            
        else:
            self.home_pi_optimizer = Adam(self.home_ac.pi.parameters(), lr=pi_lr)
        self.home_q_optimizer = Adam(self.home_q_params, lr=q_lr)
        self.has_rivals = hasattr(self, "away_ac")
        self.free_play = self.free_play*self.has_rivals
        #setup saver:
        self.logger.setup_pytorch_saver(self.home_ac)
        #setup saver:
        if self.has_rivals:
            self.logger.setup_pytorch_saver(self.away_ac)
        
        

    def create_team(self, home_or_away, n_players, actor_critic, ac_kwargs, actor_state_dict=None):
        # Create actor-critic module and target networks for each team:
        # create actor critic agent for home team

        polyak = self.training_param_dict['polyak']
        ac = actor_critic(home_or_away, self.home, self.away, self.env.observation_space.shape[0], self.env.action_space, self.loss_param_dict, polyak)
        if actor_state_dict:
            model_dict = ac.pi[0].state_dict()
            pretrained_dict = {k: v for k, v in torch.load(actor_state_dict).pi[0].state_dict().items() if k in model_dict and v.shape==model_dict[k].shape}
            model_dict.update(pretrained_dict)
            setattr(ac, 'total_delay', self.training_param_dict["start_steps"])           
            for i in range(len(ac.pi)):
                ac.pi[i].load_state_dict(model_dict)
                ac.pi[i].train()
                for k, v in ac.pi[i].named_parameters():
                    if 'propEncoder' in k:
                        v.requires_grad=False
            
        ac.train()
        ac = ac.cuda()
        ac_targ = deepcopy(ac)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False
        
        # List of parameters for both Q-networks (save this for convenience)
        q_params = list(itertools.chain(ac.q1.parameters(), ac.q2.parameters()))

        ## creation of replay buffers, we need n_players replay buffers + 1 replay buffer for the critics:
        # Experience buffer:
        obs_mat = (n_players, self.obs_dim[0])
        act_mat = (n_players,self.act_dim)
        critic_buffer = ReplayBuffer(obs_mat, act_dim=act_mat, size=self.replay_size, rew_dim=n_players)
        var_counts = list(count_vars(module) for module in [*ac.pi, ac.q1, ac.q2])
        return ac, ac_targ, q_params, critic_buffer, var_counts
    
    def update(self, data_home, data_away, timer):
        policy_delay = self.training_param_dict['policy_delay']
        polyak = self.training_param_dict['polyak']
        self.home_ac.update(data_home, self.home_q_optimizer, self.home_pi_optimizer, self.home_ac_targ,\
                            timer, self.logger, self.home_q_params, policy_delay)
        if self.free_play: 
            self.away_ac.update(data_away, self.away_q_optimizer, self.away_pi_optimizer, self.away_ac_targ,\
                            timer, self.logger, self.away_q_params, policy_delay)
        if (timer % policy_delay) == 0 :
            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                # update policy:                
                for p, p_targ in zip(self.home_ac.parameters(), self.home_ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.copy_((1-polyak) * p.data + polyak * p_targ.data)
                if self.free_play:
                    for p, p_targ in zip(self.away_ac.parameters(), self.away_ac_targ.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        p_targ.data.copy_((1-polyak) * p.data + polyak * p_targ.data)
        


    def get_action(self, o, noise_scale):
        act_lim = self.loss_param_dict['act_limit']
        actions = [self.home_ac.act(torch.as_tensor(o[0][None, ...], dtype=torch.float32).cuda()).detach().cpu().numpy()]
        actions_away= []
        if self.has_rivals:
            if self.free_play:
                actions_away = self.away_ac.act(torch.as_tensor(o[1][None, ...], dtype=torch.float32).cuda()).detach().cpu().numpy()
            else: 
                obs = o[1][None, ...]
                actions_away = self.away_ac(torch.as_tensor(obs, dtype=torch.float32).cuda()).detach().cpu().numpy()
        actions = np.concatenate(actions +  actions_away, axis=1)
        actions += noise_scale*np.random.randn(*actions.shape)
        return np.clip(actions, -act_lim, act_lim)
    
    def test_agent(self):
        succes_rate = 0
        mean_n_pass = 0
        num_test_episodes = self.training_param_dict["num_test_episodes"]
        max_ep_len = self.training_param_dict["max_ep_len"]
        vel_to_ball = [[] for j in range(self.home)]
        for j in range(num_test_episodes):
            o, ep_ret, d, ep_len = self.test_env.reset(),np.array([0]*(self.home + self.away), dtype='float32'), False,0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                actions = self.get_action(o, 0)
                o, r, d, _  = self.test_env.step([actions[0, k, :] for k in range(self.home+self.away)])
                mean_n_pass += float(np.any([o['stats_i_received_pass_10m'] or o['stats_i_received_pass_15m'] for o in self.test_env.timestep.observation[:self.home]]))
                [vel_to_ball[j].append(self.test_env.timestep.observation[j]['stats_vel_to_ball']) for j in range(self.home)]
                ep_ret += r
                ep_len += 1
            if (ep_len <= max_ep_len) and (self.test_env.timestep.reward[0] > 0):
                succes_rate += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        succes_rate /= num_test_episodes
        mean_n_pass /= num_test_episodes

        ep_ret_dict = {}
        for i in range(self.home):
            ep_ret_dict[f"TestEpStatsVelToBall_P{i + 1}"] = np.mean(vel_to_ball[i])

        self.logger.store(**ep_ret_dict, TestEpLen=ep_len)

        return succes_rate, mean_n_pass


    def train_agents(self):
        epochs = self.training_param_dict["epochs"]
        save_epochs = epochs - int(epochs*0.2)
        steps_per_epoch = self.training_param_dict["steps_per_epoch"]
        save_freq = self.training_param_dict["save_freq"]
        total_steps = steps_per_epoch*epochs
        start_steps = self.training_param_dict["start_steps"]
        start_time = time.time()
        max_ep_len = self.training_param_dict["max_ep_len"]
        o, ep_ret, ep_len = self.env.reset(), np.array([0]*(self.home + self.away), dtype='float32'), 0

        for t in tqdm(range(total_steps)):
            self.home_ac.actual_delay = t
            if hasattr(self, "away_ac"): 
                self.away_ac.actual_delay = t
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise).          
            if t > start_steps:
                with torch.no_grad():
                    a = self.get_action(o, self.act_noise)
                    a = [a[0, i, :] for i in range(self.home+self.away)]
            else:
                a = [self.env.action_space.sample() for _ in range(self.home+self.away)]
            
            
            # step in the env:
            o2, r, d, _ = self.env.step(a)
            ep_ret += np.array(r)
            ep_len += 1

            d = False if (ep_len == max_ep_len) and not d else d

            # store in buffer: 
            self.home_critic_buffer.store(o[0][None, ...], np.vstack(a[:self.home])[None, ...] , np.hstack(r[:self.home])[None, ...], o2[0][None, ...], d)
            if self.free_play:
                self.away_critic_buffer.store(o[1][None, ...], a[self.home:], r[self.home:], o2[1], d)

            #update observation: 
            o = o2

            if d or (ep_len == max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), np.array([0]*(self.home+self.away), dtype='float32'), 0
                # switch rivals if neccesary
                if not self.free_play and self.away>0:
                    self.rivals.append(self.away_ac)
                    self.away_ac = self.rivals.pop(0)

            
            # Update handling
            if (t+1) >= self.training_param_dict[ "update_after"] and (t+1) % self.training_param_dict["update_every"] == 0:
                for j in range(self.training_param_dict["update_every"]):
                    # sample from home buffers: 
                    batch_q_home = self.home_critic_buffer.sample_batch(self.training_param_dict["batch_size"])
                    batch_q_away = None
                    if self.free_play:
                        batch_q_away = self.away_critic_buffer.sample_batch(self.training_param_dict["batch_size"])
                    self.update(batch_q_home, batch_q_away, timer=j)

            if (t+1)% steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch

                # Test the performance of the deterministic version of the agent.
                with torch.no_grad():
                    succes_rate, mean_n_pass= self.test_agent()

                if (epoch% save_freq ==0) or (epoch>save_epochs) and succes_rate>=0.9:
                    self.logger.save_state({'env': self.env}, t)
                    
                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('Success rate', succes_rate)
                self.logger.log_tabular("passAverage", mean_n_pass)
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