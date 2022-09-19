from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Agent(object):

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 learning_rate: float,
                 gamma: float,
                 device: Any = None):
        self.gamma = gamma
        self.model = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        # 折扣因子
        self.gamma = gamma
        self.device = device

    def sample(self, obs: np.ndarray) -> Tuple[int, float]:
        obs = torch.FloatTensor([obs]).to(self.device)
        prob = self.model(obs)
        action_dist = Categorical(prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def predict(self, obs) -> int:
        obs = torch.FloatTensor([obs]).to(self.device)
        # 根据动作概率选择概率最高的动作
        select_action = self.model(obs).argmax().item()
        return select_action

    def learn(self, obs: torch.tensor, action: torch.tensor,
              reward: torch.tensor) -> torch.tensor:
        """Update model with policy gradient algorithm.

        Args:
            obs (torch.tensor): shape of (batch_size, obs_dim)
            action (torch.tensor): shape of (batch_size, 1)
            reward (torch.tensor): shape of (batch_size, 1)

        Returns:
            loss (torch.tensor): shape of (1)
        """
        prob = self.model(obs)
        log_prob = Categorical(prob).log_prob(action)
        loss = torch.mean(-1 * log_prob * reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def update(self, obs_list: list, action_list: list,
               reward_list: list) -> None:
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([obs_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.model(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降

    def updat_with_baseline(self, log_probs: list, returns: list) -> float:
        baseline = np.mean(returns)
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * (R - baseline))

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).mean()
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
        return loss.item()
