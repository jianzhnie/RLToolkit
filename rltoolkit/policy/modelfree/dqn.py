'''
Author: jianzhnie
Date: 2022-09-02 12:53:53
LastEditors: jianzhnie
LastEditTime: 2022-09-04 22:12:51
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import copy

import torch
import torch.optim as optim

from rltoolkit.policy.base_policy import Algorithm
from rltoolkit.utils.utils import check_model_method

__all__ = ['DQN']


class DQN(Algorithm):

    def __init__(self, model, gamma=None, lr=None, device='cpu'):
        """DQN algorithm.

        Args:
            model (Model): model defining forward network of Q function.
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        # checks
        check_model_method(model, 'forward', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)
        self.model.to(device)
        self.target_model.to(device)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = torch.nn.MSELoss()
        self.smoothl1_loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, obs):
        """use value model self.model to predict the action value."""
        pred_q = self.model(obs)
        return pred_q

    def learn(self, obs, action, reward, next_obs, terminal):
        """update value model self.model with DQN algorithm."""
        # 当前状态 Q 值
        pred_value = self.model(obs).gather(1, action)
        with torch.no_grad():
            # 下一个状态 Q 值
            next_q_value = self.target_model(next_obs).max(1, keepdim=True)[0]
            # 计算 TD-Target
            target = reward + (1 - terminal) * self.gamma * next_q_value

        # TD误差目标
        loss = self.mse_loss(pred_value, target)
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        loss.backward()
        # 反向传播更新参数
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        # 更新目标网络
        self.model.sync_weights_to(self.target_model)
