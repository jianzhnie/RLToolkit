'''
Author: jianzhnie
Date: 2022-09-02 14:44:14
LastEditors: jianzhnie
LastEditTime: 2022-09-02 14:44:18
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rltoolkit.core.algorithm import Algorithm
from rltoolkit.utils.utils import check_model_method

__all__ = ['A2C']


class A2C(Algorithm):

    def __init__(self, model, config):
        # checks
        assert isinstance(config['vf_loss_coeff'], (int, float))
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'policy_and_value', self.__class__.__name__)

        self.model = model
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config['learning_rate'])
        self.config = config

    def learn(self, obs, actions, advantages, target_values, lr,
              entropy_coeff):
        logits = self.model.policy(obs)
        act_dim = logits.shape[-1]
        actions_onehot = F.one_hot(actions, act_dim)
        actions_log_probs = torch.sum(
            F.log_softmax(logits, dim=1) * actions_onehot, dim=-1)
        # The policy gradient loss
        pi_loss = -1.0 * torch.sum(actions_log_probs * advantages)

        # The value function loss
        values = self.model.value(obs)
        delta = values - target_values
        vf_loss = 0.5 * torch.sum(torch.square(delta))

        policy_distri = Categorical(logits=logits)
        # The entropy loss (We want to maximize entropy, so entropy_ceoff < 0)
        policy_entropy = policy_distri.entropy()
        entropy = torch.sum(policy_entropy)

        total_loss = pi_loss + vf_loss * self.vf_loss_coeff + entropy * entropy_coeff

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        total_loss.backward()
        # clip the grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=40.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss, pi_loss, vf_loss, entropy

    def sample(self, obs):
        logits, values = self.model.policy_and_value(obs)
        sample_actions = Categorical(logits=logits).sample().long()
        return sample_actions, values

    def prob_and_value(self, obs):
        logits, values = self.model.policy_and_value(obs)
        probs = F.softmax(logits, dim=1)
        return probs, values

    def predict(self, obs):
        prob = self.model.policy(obs)
        _, predict_actions = prob.max(-1)
        return predict_actions

    def value(self, obs):
        values = self.model.value(obs)
        return values
