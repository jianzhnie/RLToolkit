from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.optim import Adam


class PolicyNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
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

    def __init__(self, obs_dim: int, hidden_dim: int):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ActorCritic(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(ActorCritic).__init__()
        """Initialize."""

        self.policy_net = PolicyNet(obs_dim, hidden_dim, action_dim)
        self.value_net = ValueNet(obs_dim, hidden_dim)

    def policy(self, x: torch.Tensor) -> torch.Tensor:
        out = self.policy_net(x)
        return out

    def value(self, x: torch.Tensor) -> torch.Tensor:
        out = self.value_net(x)
        return out


class Agent(object):
    """Agent interacting with environment. The “Critic” estimates the value
    function. This could be the action-value (the Q value) or state-value (the
    V value). The “Actor” updates the policy distribution in the direction
    suggested by the Critic (such as with policy gradients).

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
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 gamma: float,
                 lmbda: float,
                 clip_param: float = 0.2,
                 target_kl: float = 0.01,
                 train_policy_iters: int = 10,
                 train_value_iters: int = 10,
                 device: Any = None):

        self.gamma = gamma
        self.lmbda = lmbda
        self.clip_param = clip_param  # PPO中截断范围的参数
        self.target_kl = target_kl
        self.train_policy_iters = train_policy_iters  # 一条序列的数据用来训练轮数
        self.train_value_iters = train_value_iters  # 一条序列的数据用来训练轮数

        # 策略网络
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = ValueNet(obs_dim, hidden_dim).to(device)
        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        # 折扣因子
        self.device = device

    def sample(self, obs: np.ndarray) -> Tuple[int, float]:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        prob = self.actor(obs)
        action_dist = Categorical(prob)
        action = action_dist.sample()
        return action.item()

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        prob = self.actor(obs)
        action_dist = Categorical(prob)
        action = action_dist.sample()
        return action.item()

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def learn(self, transition_dict) -> Tuple[float, float]:
        """Update the model by TD actor-critic."""
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

        pred_value = self.critic(obs)
        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_obs) * (1 - dones)
        # 时序差分误差
        td_delta = td_target - pred_value

        advantage = self.compute_advantage(self.gamma, self.lmbda,
                                           td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(obs).gather(1, actions)).detach()
        old_action_dists = Categorical(self.actor(obs).detach())

        for _ in range(self.train_policy_iters):
            log_probs = torch.log(self.actor(obs).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            # 截断
            surr2 = torch.clamp(ratio, 1 - self.clip_param,
                                1 + self.clip_param) * advantage
            # PPO损失函数
            policy_loss = torch.mean(-torch.min(surr1, surr2))

            # K-L dist
            new_action_dists = Categorical(self.actor(obs))
            # A sample estimate for KL-divergence, easy to compute
            # approx_kl = (old_log_probs - log_probs).mean()
            kl_div = torch.mean(
                kl_divergence(old_action_dists, new_action_dists))
            # Early stopping at step i due to reaching max kl
            if kl_div > 1.5 * self.target_kl:
                print(
                    'Early stopping, due to current kl_div: %3f reaching max kl %3f'
                    % (kl_div, self.target_kl))
                break
            # update policy
            self.actor_optimizer.zero_grad()
            policy_loss.backward()  # 计算策略网络的梯度
            self.actor_optimizer.step()  # 更新策略网络的参数

        for _ in range(self.train_value_iters):
            # value loss
            value_loss = F.mse_loss(self.critic(obs), td_target.detach())
            # update value
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        return policy_loss.item(), value_loss.item()
