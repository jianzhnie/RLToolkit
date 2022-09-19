from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam


class PolicyNet(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class ValueNet(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ActorCritic(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int,
                 action_dim: int) -> None:
        super(ActorCritic).__init__()
        """Initialize."""

        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.value_net = ValueNet(state_dim, hidden_dim)

    def policy(self, x: torch.Tensor) -> torch.Tensor:
        out = self.policy_net(x)
        return out

    def value(self, x: torch.Tensor) -> torch.Tensor:
        out = self.value_net(x)
        return out


class Agent(object):
    """A2CAgent interacting with environment.

    Atribute:
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 gamma: float,
                 entropy_weight: float,
                 device: Any = None):

        self.gamma = gamma
        self.entropy_weight = entropy_weight

        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        # 折扣因子
        self.gamma = gamma
        self.device = device

    def sample(self, obs: np.ndarray) -> Tuple[int, float]:
        obs = torch.tensor([obs], dtype=torch.float32).to(self.device)
        prob = self.actor(obs)
        action_dist = Categorical(prob)
        action = action_dist.sample()
        return action.item()

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        # 根据动作概率选择概率最高的动作
        select_action = self.critic(obs).argmax().item()
        return select_action

    def learn(self, transition_dict) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        obs = torch.tensor(
            transition_dict['obs'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype=torch.float).view(-1, 1).to(self.device)
        next_obs = torch.tensor(
            transition_dict['next_obs'], dtype=torch.float).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'],
            dtype=torch.float).view(-1, 1).to(self.device)

        log_probs = torch.log(self.actor(obs).gather(1, actions))

        pred_value = self.critic(obs)
        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_obs) * (1 - dones)
        # 均方误差损失函数
        value_loss = F.mse_loss(pred_value, td_target.detach())

        # 时序差分误差
        td_delta = td_target - pred_value
        policy_loss = torch.mean(-log_probs * td_delta.detach())

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()  # 计算策略网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数

        return policy_loss.item(), value_loss.item()

    def learn_a2c(self, transition_dict) -> None:
        """Update the model by gradient descent."""
        obs = torch.tensor(
            transition_dict['obs'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype=torch.float).view(-1, 1).to(self.device)
        next_obs = torch.tensor(
            transition_dict['next_obs'], dtype=torch.float).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'],
            dtype=torch.float).view(-1, 1).to(self.device)

        log_probs = torch.log(self.actor(obs).gather(1, actions))

        pred_value = self.critic(obs)
        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_obs) * (1 - dones)
        # 均方误差损失函数
        value_loss = F.smooth_l1_loss(pred_value, td_target.detach())

        # advantage = Q_t - V(s_t)
        advantage = (td_target - pred_value).detach()  # not backpropagated
        policy_loss = -advantage * log_probs
        policy_loss += -self.entropy_weight * log_probs  # entropy maximization
        policy_loss = torch.mean(policy_loss)

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        return policy_loss.item(), value_loss.item()
