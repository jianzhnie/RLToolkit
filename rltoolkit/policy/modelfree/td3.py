'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

from copy import deepcopy

import torch
import torch.nn.functional as F

from rltoolkit.models.utils import check_model_method
from rltoolkit.policy.base_policy import Algorithm

__all__ = ['TD3']


class TD3(Algorithm):

    def __init__(
        self,
        model,
        gamma=None,
        tau=None,
        actor_lr=None,
        critic_lr=None,
        policy_noise=0.2,  # Noise added to target policy during critic update
        noise_clip=0.5,  # Range to clip target policy noise
        policy_freq=2):  # Frequency of delayed policy updates
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'Q1', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.target_model = deepcopy(model).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)

        self.total_it = 0

    def predict(self, obs):
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        self.total_it += 1
        self._critic_learn(obs, action, reward, next_obs, terminal)
        if self.total_it % self.policy_freq == 0:
            self._actor_learn(obs)

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)

            next_action = (self.target_model.policy(next_obs) + noise).clamp(
                -1, 1)

            target_Q1, target_Q2 = self.target_model.value(
                next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - terminal) * self.gamma * target_Q

        current_Q1, current_Q2 = self.model.value(obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _actor_learn(self, obs):
        actor_loss = -self.model.Q1(obs, self.model.policy(obs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.sync_target()

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for param, target_param in zip(self.model.parameters(),
                                       self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)
