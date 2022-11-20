import copy
from typing import Any, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class GaussianNoise(object):
    """Gaussian Noise.

    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        action_dim: int,
        min_sigma: float = 1.0,
        max_sigma: float = 1.0,
        decay_period: int = 1000000,
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period)
        return np.random.normal(0, sigma, size=self.action_dim)


class PolicyNet(nn.Module):

    def __init__(self,
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 init_w: float = 3e-3):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # 连续动作空间，利用 tanh() 函数将特征映射到 [-1, 1],
        # 然后通过变换，得到 [low, high] 的输出
        out = torch.tanh(x)
        return out


class Critic(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        init_w: float = 3e-3,
    ):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

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

    Attribute:
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
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 action_bound: float = 1.0,
                 exploration_noise: float = 0.1,
                 target_policy_noise: float = 0.2,
                 target_policy_noise_clip: float = 0.5,
                 initial_random_steps: int = int(1e3),
                 policy_update_freq: int = 2,
                 device: Any = None):

        self.env = env
        self.gamma = gamma
        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound
        # 目标网络软更新参数
        self.tau = tau
        self.global_update_step = 0
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq

        # noise
        self.exploration_noise = GaussianNoise(action_dim, exploration_noise,
                                               exploration_noise)
        self.target_policy_noise = GaussianNoise(action_dim,
                                                 target_policy_noise,
                                                 target_policy_noise)
        self.target_policy_noise_clip = target_policy_noise_clip

        # 策略网络
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic1 = Critic(obs_dim, hidden_dim, action_dim).to(device)
        self.critic2 = Critic(obs_dim, hidden_dim, action_dim).to(device)

        # target network
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        # concat critic parameters to use one optim
        self.critic_parameters = list(self.critic1.parameters()) + list(
            self.critic2.parameters())

        self.critic_optimizer = Adam(self.critic_parameters, lr=critic_lr)
        self.device = device

    def sample(self, obs: np.ndarray):
        if self.global_update_step < self.initial_random_steps:
            selected_action = self.env.action_space.sample()
        else:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            selected_action = self.actor(obs).detach().cpu().numpy()
            # add noise for exploration during training
            noise = self.exploration_noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
            selected_action *= self.action_bound
        selected_action = selected_action.flatten()
        return selected_action

    def predict(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        selected_action = self.actor(obs).detach().cpu().numpy().flatten()
        selected_action *= self.action_bound
        return selected_action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def learn(self, obs: torch.Tensor, action: torch.Tensor,
              reward: torch.Tensor, next_obs: torch.Tensor,
              terminal: torch.Tensor) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""

        # get actions with noise
        noise = torch.FloatTensor(self.target_policy_noise.sample()).to(
            self.device)
        clipped_noise = torch.clamp(noise, -self.target_policy_noise_clip,
                                    self.target_policy_noise_clip)

        next_pi_tgt = self.target_actor(next_obs)
        next_pi_tgt = (next_pi_tgt + clipped_noise).clamp(-1.0, 1.0)
        next_pi_tgt *= self.action_bound

        # pred q value
        # Prediction Q1(s,𝜇(s)), Q1(s,a), Q2(s,a)
        pred_values1 = self.critic1(obs, action)
        pred_values2 = self.critic2(obs, action)

        # Min Double-Q: min(Q1‾(s',𝜇(s')), Q2‾(s',𝜇(s')))
        next_q1_pi_tgt = self.target_critic1(next_obs, next_pi_tgt)
        next_q2_pi_tgt = self.target_critic2(next_obs, next_pi_tgt)
        min_next_q_pi_tgt = torch.min(next_q1_pi_tgt, next_q2_pi_tgt)

        # 时序差分目标
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        td_target = reward + self.gamma * min_next_q_pi_tgt * (1 - terminal)

        # 均方误差损失函数
        critic1_loss = F.mse_loss(pred_values1, td_target.detach())
        critic2_loss = F.mse_loss(pred_values2, td_target.detach())

        # update value network
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # cal policy loss
        if self.global_update_step % self.policy_update_freq == 0:

            pi = self.actor(obs)
            q1_pi = self.critic1(obs, pi)
            actor_loss = -torch.mean(q1_pi)

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新策略网络
            self.soft_update(self.actor, self.target_actor)
            # 软更新价值网络
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)

        else:
            actor_loss = torch.zeros(1)

        self.global_update_step += 1
        return actor_loss.item(), critic1_loss.item()
