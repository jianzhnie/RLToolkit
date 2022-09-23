import copy
from typing import Any, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
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

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
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
                 target_entropy: float,
                 tau: float,
                 gamma: float,
                 initial_random_steps: int = int(1e4),
                 device: Any = None):

        self.env = env
        self.action_dim = action_dim
        self.global_update_step = 0
        # 目标熵的大小
        self.target_entropy = target_entropy
        # 折扣因子
        self.gamma = gamma
        # 目标网络软更新参数
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        # 策略网络
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic1 = ValueNet(obs_dim, hidden_dim, action_dim).to(device)
        self.critic2 = ValueNet(obs_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(
            np.log(0.01), dtype=torch.float, device=device)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

        self.device = device

    def sample(self, obs: np.ndarray):
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.global_update_step < self.initial_random_steps:
            action = self.env.action_space.sample()
        else:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            probs = self.actor(obs)
            action_dist = Categorical(probs)
            action = action_dist.sample().item()
        return action

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        probs = self.actor(obs)
        action_dist = Categorical(probs)
        action = action_dist.sample().item()
        return action

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_obs, terminal):
        next_probs = self.actor(next_obs)
        prob_dist = Categorical(next_probs)
        entropy = prob_dist.entropy()
        q1_value = self.target_critic1(next_obs)
        q2_value = self.target_critic2(next_obs)
        min_qvalue = torch.sum(
            next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - terminal)
        return td_target

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
        actions = torch.LongTensor(action.reshape(-1, 1)).to(device)
        rewards = torch.FloatTensor(reward.reshape(-1, 1)).to(device)
        terminal = torch.FloatTensor(terminal.reshape(-1, 1)).to(device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_obs, terminal)
        critic1_q_values = self.critic1(obs).gather(1, actions)
        critic1_loss = F.mse_loss(critic1_q_values, td_target.detach())
        critic2_q_values = self.critic2(obs).gather(1, actions)
        critic2_loss = F.mse_loss(critic2_q_values, td_target.detach())

        # critic1_loss backward
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic1_loss backward
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新策略网络
        probs = self.actor(obs)
        probs_dist = Categorical(probs)
        # 直接根据概率计算熵
        entropy = probs_dist.entropy()
        q1_value = self.critic1(obs)
        q2_value = self.critic2(obs)
        min_qvalue = torch.sum(
            probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        # backward
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        self.global_update_step += 1

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()
