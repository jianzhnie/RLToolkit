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
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        prob = self.model(obs)
        action_dist = Categorical(prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        # 根据动作概率选择概率最高的动作
        select_action = self.model(obs).argmax().item()
        return select_action

    def learn(self, log_probs: list, returns: list) -> None:
        """Update model with policy gradient algorithm.

        Returns:
            loss (torch.tensor): shape of (1)
        """
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        loss = torch.cat(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
        return loss

    def learn_with_baseline(self, log_probs: list, returns: list) -> float:
        baseline = np.mean(returns)
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * (R - baseline))

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
        return loss.item()
