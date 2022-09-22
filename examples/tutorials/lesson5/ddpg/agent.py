import copy
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class PolicyNet(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 action_bound: float):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # 连续动作空间，利用 tanh() 函数将特征映射到 [-1, 1],
        # 然后通过变换，得到 [low, high] 的输出
        out = torch.tanh(x) * self.action_bound
        return out


class ValueNet(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)
        x = self.fc1(cat)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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
                 state_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 action_bound: float,
                 sigma: float,
                 tau: float,
                 gamma: float,
                 device: Any = None):

        self.action_dim = action_dim
        self.gamma = gamma
        # 高斯噪声的标准差,均值直接设为0
        self.sigma = sigma
        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound
        # 目标网络软更新参数
        self.tau = tau

        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim,
                               action_bound).to(device)
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)

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
        action = self.actor(obs).item()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def predict(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        select_action = self.actor(obs).cpu().detach().numpy().flatten()
        return select_action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def learn(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_obs: np.ndarray,
              terminal: np.ndarray) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""

        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        actions = torch.FloatTensor(action.reshape(-1, 1)).to(device)
        rewards = torch.FloatTensor(reward.reshape(-1, 1)).to(device)
        terminal = torch.FloatTensor(terminal.reshape(-1, 1)).to(device)

        next_actions = self.target_actor(next_obs)
        next_q_values = self.target_critic(next_obs, next_actions)

        # 时序差分目标
        q_targets = rewards + self.gamma * next_q_values * (1 - terminal)
        # 均方误差损失函数
        value_loss = F.mse_loss(self.critic(obs, actions), q_targets)
        # update value network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # cal policy loss
        policy_loss = -torch.mean(self.critic(obs, self.actor(obs)))
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络
        self.soft_update(self.critic, self.target_critic)

        return policy_loss.item(), value_loss.item()
