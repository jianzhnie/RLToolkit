'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from rltoolkit.models import Model


class MujocoModel(Model):
    """Model of DDPG:  defines an Actor/policy network given obs as input,

    & a Critic/value network given obs and action as input.
    """

    def __init__(self, obs_dim, action_dim):
        super(MujocoModel, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(Model):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, obs):
        a = F.relu(self.l1(obs))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class Critic(Model):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, obs, action):
        q = F.relu(self.l1(obs))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)
