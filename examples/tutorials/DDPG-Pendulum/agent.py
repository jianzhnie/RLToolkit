import copy
from typing import Any, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from rltoolkit.models.ounoise import OUNoise
from rltoolkit.models.utils import soft_target_update


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


class ValueNet(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        init_w: float = 3e-3,
    ):
        super(ValueNet, self).__init__()
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
                 action_bound: float,
                 ou_noise_theta: float,
                 ou_noise_sigma: float,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 update_target_step: int = 1,
                 device: Any = None):

        self.env = env
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        # 目标网络软更新参数
        self.tau = tau
        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound
        self.global_update_step = 0
        self.update_target_step = update_target_step

        # noise
        self.noise = OUNoise(
            mu=0.0, sigma=ou_noise_sigma, theta=ou_noise_theta)

        # 策略网络
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = ValueNet(obs_dim, hidden_dim, action_dim).to(device)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        # 折扣因子
        self.device = device

    def sample(self, obs: np.ndarray):

        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        selected_action = self.actor(obs).detach().cpu().numpy()
        # add noise for exploration during training
        noise = self.noise(selected_action.shape)
        selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        selected_action *= self.action_bound
        selected_action = selected_action.flatten()
        return selected_action

    def predict(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        selected_action = self.actor(obs).detach().cpu().numpy().flatten()
        selected_action *= self.action_bound
        return selected_action

    def learn(self, obs: torch.Tensor, action: torch.Tensor,
              reward: torch.Tensor, next_obs: torch.Tensor,
              terminal: torch.Tensor) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""

        pred_q_values = self.critic(obs, action)
        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            next_q_values = self.target_critic(next_obs, next_actions)

        # 时序差分目标
        q_targets = reward + self.gamma * next_q_values * (1 - terminal)
        # 均方误差损失函数
        value_loss = F.mse_loss(pred_q_values, q_targets)
        # update value network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # cal policy loss
        # For the policy function, our objective is to maximize the expected return
        # To calculate the policy loss, we take the derivative of the objective function with respect to the policy parameter.
        # Keep in mind that the actor (policy) function is differentiable, so we have to apply the chain rule.
        policy_loss = -torch.mean(self.critic(obs, self.actor(obs)))
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.global_update_step % self.update_target_step == 0:
            # 软更新策略网络
            soft_target_update(self.actor, self.target_actor, tau=self.tau)
            # 软更新价值网络
            soft_target_update(self.critic, self.target_critic, tau=self.tau)

        self.global_update_step += 1
        return policy_loss.item(), value_loss.item()
