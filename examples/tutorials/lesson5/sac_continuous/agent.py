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
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super(PolicyNet, self).__init__()
        # set the log std range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # set log_std layer
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)

        # get mean
        mu = torch.tanh(self.mu_layer(out))
        # get std
        log_std = self.log_std_layer(out).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (
                                                log_std + 1)
        std = torch.exp(log_std)

        # sample actions
        dist = Normal(mu, std)
        z = dist.rsample()
        # normalize action and log_prob
        # see appendix C of [2]
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


class CriticQ(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(CriticQ, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)

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
                 tau: float,
                 gamma: float,
                 alpha: float = 0.2,
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
        self.initial_random_steps = initial_random_steps

        # 策略网络
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic1 = CriticQ(obs_dim, hidden_dim, action_dim).to(device)
        self.critic2 = CriticQ(obs_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        # automatic entropy tuning
        self.target_entropy = -np.prod((action_dim, )).item()  # heuristic
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
            action, probs = self.actor(obs)
            action = action.detach().cpu().numpy().flatten()
        return action

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, probs = self.actor(obs)
        action = action.detach().cpu().numpy()
        return action

    # Set up function for computing SAC Q-losses
    def compute_critic_loss(self, obs: torch.Tensor, action: torch.Tensor,
                            reward: torch.Tensor, next_obs: torch.Tensor,
                            terminal: torch.Tensor) -> torch.Tensor:

        pred_q1_value = self.critic1(obs, action)
        pred_q2_value = self.critic2(obs, action)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, next_action_logp = self.actor(next_obs)
            # Target Q-values
            q1_target = self.target_critic1(next_obs, next_action)
            q2_target = self.target_critic2(next_obs, next_action)

            q_target = torch.min(q1_target, q2_target)
            td_target = reward + self.gamma * (
                q_target - self.alpha * next_action_logp) * (1 - terminal)

        # MSE loss against Bellman backup
        q1_loss = F.mse_loss(pred_q1_value, td_target.detach())
        q2_loss = F.mse_loss(pred_q2_value, td_target.detach())
        q_loss = q1_loss + q2_loss

        return q_loss

    # Set up function for computing SAC pi loss
    def compute_actor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        next_action, log_prob = self.actor(obs)
        q1_pi = self.critic1(obs, next_action)
        q2_pi = self.critic2(obs, next_action)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        actor_loss = (self.alpha * log_prob - q_pi).mean()
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

        new_action, log_prob = self.actor(obs)

        # train alpha (dual problem)
        alpha_loss = (-self.log_alpha.exp() *
                      (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()  # used for the actor loss calculation

        pred_q1_value = self.critic1(obs, actions)
        pred_q2_value = self.critic2(obs, actions)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, next_action_logp = self.actor(next_obs)
            # Target Q-values
            q1_target = self.target_critic1(next_obs, next_action)
            q2_target = self.target_critic2(next_obs, next_action)

            q_target = torch.min(q1_target, q2_target)
            td_target = rewards + self.gamma * (
                q_target - self.alpha * next_action_logp) * (1 - terminal)

        # MSE loss against Bellman backup
        q1_loss = F.mse_loss(pred_q1_value, td_target.detach())
        q2_loss = F.mse_loss(pred_q2_value, td_target.detach())
        q_loss = q1_loss + q2_loss

        # actor loss
        q1_pi = self.critic1(obs, new_action)
        q2_pi = self.critic2(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy-regularized policy loss
        actor_loss = (alpha * log_prob - q_pi).mean()

        #  train Q functions
        # First run one gradient descent step for Q1 and Q2
        self.critic1.zero_grad()
        # q_loss = self.compute_critic_loss(obs, actions, rewards, next_obs,
        #                                   terminal)
        q_loss.backward()
        self.critic1_optimizer.step()

        # train actor
        self.actor.zero_grad()
        # actor_loss = self.compute_actor_loss(obs)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        self.global_update_step += 1

        return actor_loss.item(), q_loss.item()
