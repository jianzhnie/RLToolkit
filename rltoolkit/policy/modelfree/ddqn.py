'''
Author: jianzhnie
Date: 2022-09-02 14:18:50
LastEditors: jianzhnie
LastEditTime: 2022-09-02 19:43:20
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import copy

import torch
import torch.optim as optim

from rltoolkit.policy.base_policy import Algorithm
from rltoolkit.utils.utils import check_model_method

__all__ = ['DDQN']


class DDQN(Algorithm):

    def __init__(self, model, gamma=None, lr=None, device='cpu'):
        """Double DQN algorithm.

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, obs):
        """use value model self.model to predict the action value."""
        pred_q = self.model(obs)
        return pred_q

    def learn(self, obs, action, reward, next_obs, terminal):
        """update value model self.model with Double DQN algorithm."""
        pred_value = self.model(obs).gather(1, action)
        # model for selection actions.
        greedy_action = self.model(next_obs).max(dim=1, keepdim=True)[1]
        with torch.no_grad():
            # target_model for evaluation.
            max_v = self.target_model(next_obs).gather(1, greedy_action)
            target = reward + (1 - terminal) * self.gamma * max_v
        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.model.sync_weights_to(self.target_model)
