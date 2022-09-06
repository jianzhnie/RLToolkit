'''
Author: jianzhnie@126.com
Date: 2022-09-01 11:28:52
LastEditors: jianzhnie
LastEditTime: 2022-09-02 19:43:16
Description:

Copyright (c) 2022 by jianzhnie jianzhnie@126.com, All Rights Reserved.
'''
import torch
import torch.optim as optim
from torch.distributions import Categorical

from rltoolkit.policy.base_policy import Algorithm
from rltoolkit.utils.utils import check_model_method

__all__ = ['PolicyGradient']


class PolicyGradient(Algorithm):

    def __init__(self, model, lr, device) -> None:

        check_model_method(model, 'forward', self.__class__.__name__)
        assert isinstance(lr, float)
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, obs):
        prob = self.model(obs)
        return prob

    def learn(self, obs, action, reward):
        prob = self.model(obs)

        log_prob = Categorical(prob).log_prob(action)

        loss = torch.mean(-1 * log_prob * reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
