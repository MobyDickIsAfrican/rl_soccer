import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

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

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class MLPAC_4_team(nn.Module):

    def __init__(self, players, observation_space, action_space, polyak=0.1, hidden_sizes=(256, 256),
                activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.polyak = polyak

        # build policy for each player in team
        self.pi = nn.ModuleList([MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)\
                                                         for _ in range(players)])
        
        # build critic: 
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return np.numpy([pi(obs).numpy() for pi in self.pi])

    def compute_q_loss(self, data, ac_targ, kwargs):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.q1(o,a)
        q2 = self.ac.q2(o,a)

        # get training params: 
        target_noise = kwargs["target_noise"]
        noise_clip = kwargs["noise_clip"]
        act_limit = kwargs["act_limit"]
        gamma = kwargs["gamma"]


        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
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
        o = data['obs']
        q1_pi = [self.q1(o[i], self.pi[i](o[i])) for i in range(len(self.pi))]
        return torch.sum([-q1_i.mean() for q1_i in q1_pi])

    # update method for the networks: 
    def update(self, team_buffer, critic_buffer, q_optim, pi_optim, ac_targ, timer, q_param,policy_delay):
        q_optim.zero_grad()
        loss_q, loss_info = self.compute_q_loss(critic_buffer)
        loss_q.backward()
        q_optim.step()

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
            
            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.pi.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)



        

