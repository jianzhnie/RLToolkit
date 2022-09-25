import copy
from typing import Any, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class PolicyNet(nn.Module):

    def __init__(self,
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 action_bound: float,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super(PolicyNet, self).__init__()

        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound
        # set the log std range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.relu = nn.ReLU()

        # set log_std layer
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x) * clip_up + (l - x) * clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        log_pi -= torch.sum(
            torch.log(
                self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6),
            dim=-1)
        return mu, pi, log_pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)

        # get mean
        mu = self.mu_layer(out)
        # get std
        log_std = self.log_std_layer(out).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (
                                                log_std + 1)
        std = torch.exp(log_std)

        # sample actions
        # https://pytorch.org/docs/stable/distributions.html#normal
        dist = Normal(mu, std)
        # Reparameterization trick (mean + std * N(0,1))
        pi = dist.rsample()
        # normalize action and log_prob
        log_pi = dist.log_prob(pi).sum(dim=-1)
        mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)

        # Make sure outputs are in correct range
        mu = mu * self.action_bound
        pi = pi * self.action_bound

        return pi, log_pi


class CriticQ(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(CriticQ, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)
        out = self.fc1(cat)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Agent(object):
    """Agent interacting with environment.

    The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value (the V value).

    The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients).

    Atribute:
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        device (torch.device): cpu / gpu
    """

    def __init__(self,
                 env: gym.Env,
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 alpha_lr: float,
                 action_bound: float,
                 tau: float,
                 gamma: float,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = False,
                 initial_random_steps: int = int(1e4),
                 device: Any = None):

        self.env = env
        self.action_dim = action_dim
        self.global_update_step = 0
        # 折扣因子
        self.gamma = gamma
        # 目标网络软更新参数
        self.tau = tau
        self.alpha = alpha
        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.initial_random_steps = initial_random_steps

        # 策略网络
        self.actor = PolicyNet(
            obs_dim, hidden_dim, action_dim,
            action_bound=self.action_bound).to(device)
        # 价值网络
        self.critic1 = CriticQ(obs_dim, hidden_dim, action_dim).to(device)
        self.critic2 = CriticQ(obs_dim, hidden_dim, action_dim).to(device)
        #  Target network
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        # Concat the Q-network parameters to use one optim
        self.Qnet_parameters = list(self.critic1.parameters()) + list(
            self.critic2.parameters())
        self.critic_optimizer = Adam(self.Qnet_parameters, lr=critic_lr)

        # automatic entropy tuning
        # If automatic entropy tuning is True,
        # initialize a target entropy, a log alpha and an alpha optimizer
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod((action_dim, )).item()  # heuristic
            # 使用alpha的log值,可以使训练结果比较稳定
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

        self.device = device

    def sample(self, obs: np.ndarray):
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.global_update_step < self.initial_random_steps:
            action = self.env.action_space.sample()
        else:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action, log_probs = self.actor(obs)
            action = action.detach().cpu().numpy()
        action = action.flatten()
        return action

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, log_probs = self.actor(obs)
        action = action.detach().cpu().numpy().flatten()
        action = action.flatten()
        return action

    # Set up function for computing SAC Q-losses
    def compute_critic_loss(self, obs: torch.Tensor, action: torch.Tensor,
                            reward: torch.Tensor, next_obs: torch.Tensor,
                            terminal: torch.Tensor) -> torch.Tensor:

        # Prediction π(a|s), logπ(a|s),
        # π(a'|s'), logπ(a'|s'),
        pi, log_pi = self.actor(obs)
        # Target actions come from *current* policy
        next_pi, next_log_pi = self.actor(next_obs)

        # Prediction Q1(s,a), Q2(s,a)
        pred_q1_value = self.critic1(obs, action)
        pred_q2_value = self.critic2(obs, action)

        # Target Q-values
        # Min Double-Q: min(Q1‾(s',π(a'|s')), Q2‾(s',π(a'|s')))
        q1_next_pi = self.target_critic1(next_obs, next_pi)
        q2_next_pi = self.target_critic2(next_obs, next_pi)
        min_q_next_pi = torch.min(q1_next_pi, q2_next_pi)

        # TD target for Q regression
        td_v_target = min_q_next_pi - self.alpha * next_log_pi
        td_q_target = reward + self.gamma * td_v_target * (1 - terminal)

        # MSE loss against Bellman backup
        q1_loss = F.mse_loss(pred_q1_value, td_q_target.detach())
        q2_loss = F.mse_loss(pred_q2_value, td_q_target.detach())
        q_loss = q1_loss + q2_loss

        return q_loss

    # Set up function for computing SAC pi loss
    def compute_actor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        # Prediction π(a|s), logπ(a|s),
        # π(a'|s'), logπ(a'|s'),
        pi, log_pi = self.actor(obs)

        # Min Double-Q: min(Q1(s,π(a|s)), Q2(s,π(a|s))),
        q1_pi = self.critic1(obs, pi)
        q2_pi = self.critic2(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        actor_loss = (self.alpha * log_pi - min_q_pi).mean()
        return actor_loss

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def learn(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_obs: np.ndarray,
              terminal: np.ndarray) -> Tuple[float, float]:

        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        actions = torch.LongTensor(action.reshape(-1, 1)).to(device)
        rewards = torch.FloatTensor(reward.reshape(-1, 1)).to(device)
        terminal = torch.FloatTensor(terminal.reshape(-1, 1)).to(device)

        # Prediction π(a|s), logπ(a|s), π(a'|s'), logπ(a'|s'),
        pi, log_pi = self.actor(obs)
        # Target actions come from *current* policy
        next_pi, next_log_pi = self.actor(next_obs)

        #  Prediction Q1(s,a), Q2(s,a)
        pred_q1_value = self.critic1(obs, actions)
        pred_q2_value = self.critic2(obs, actions)

        # Min Double-Q: min(Q1(s,π(a|s)), Q2(s,π(a|s))),
        q1_pi = self.critic1(obs, pi)
        q2_pi = self.critic2(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Target Q-values
        # Min Double-Q: min(Q1‾(s',π(a'|s')), Q2‾(s',π(a'|s')))
        q1_next_pi = self.target_critic1(next_obs, next_pi)
        q2_next_pi = self.target_critic2(next_obs, next_pi)
        min_q_next_pi = torch.min(q1_next_pi, q2_next_pi)

        # TD target for Q regression
        td_v_target = min_q_next_pi - self.alpha * next_log_pi
        td_q_target = rewards + self.gamma * td_v_target * (1 - terminal)

        # Entropy-regularized policy loss
        policy_loss = torch.mean(self.alpha * log_pi - min_q_pi)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # MSE loss against Bellman backup
        q1_loss = F.mse_loss(pred_q1_value, td_q_target.detach())
        q2_loss = F.mse_loss(pred_q2_value, td_q_target.detach())
        q_loss = q1_loss + q2_loss

        # Update two Q-network parameter
        # First run one gradient descent step for Q1 and Q2
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # If automatic entropy tuning is True, update alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # used for the actor loss calculation
            self.alpha = self.log_alpha.exp()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        self.global_update_step += 1

        return policy_loss.item(), q_loss.item()
