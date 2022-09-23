import copy
from typing import Any, Tuple

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
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 alpha_lr: float,
                 target_entropy: float,
                 tau: float,
                 gamma: float,
                 device: Any = None):

        self.action_dim = action_dim
        self.global_update_step = 0
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        # 目标网络软更新参数
        self.tau = tau

        # 策略网络
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.ceritic1 = ValueNet(obs_dim, hidden_dim, action_dim).to(device)
        self.ceritic2 = ValueNet(obs_dim, hidden_dim, action_dim).to(device)
        self.target_ceritic1 = copy.deepcopy(self.ceritic1)
        self.target_ceritic2 = copy.deepcopy(self.ceritic2)

        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.ceritic1_optimizer = Adam(
            self.ceritic1.parameters(), lr=critic_lr)
        self.ceritic2_optimizer = torch.optim.Adam(
            self.ceritic2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        # 折扣因子
        self.device = device

    def sample(self, obs: np.ndarray):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        probs = self.actor(obs)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        # 根据动作概率选择概率最高的动作
        select_action = self.actor(obs).argmax().item()
        return select_action

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_obs, terminal):
        next_probs = self.actor(next_obs)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_ceritic1(next_obs)
        q2_value = self.target_ceritic2(next_obs)
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
        ceritic1_q_values = self.ceritic1(obs).gather(1, actions)
        ceritic1_loss = F.mse_loss(ceritic1_q_values, td_target.detach())
        ceritic2_q_values = self.ceritic2(obs).gather(1, actions)
        ceritic2_loss = F.mse_loss(ceritic2_q_values, td_target.detach())

        # ceritic1_loss backward
        self.ceritic1_optimizer.zero_grad()
        ceritic1_loss.backward()
        self.ceritic1_optimizer.step()
        # ceritic1_loss backward
        self.ceritic2_optimizer.zero_grad()
        ceritic2_loss.backward()
        self.ceritic2_optimizer.step()

        # 更新策略网络
        probs = self.actor(obs)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.ceritic1(obs)
        q2_value = self.ceritic2(obs)
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

        self.soft_update(self.ceritic1, self.target_ceritic1)
        self.soft_update(self.ceritic2, self.target_ceritic2)

        return actor_loss.item(), ceritic1_loss.item(), ceritic2_loss.item()
