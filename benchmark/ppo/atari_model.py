'''
Author: jianzhnie@126.com
Date: 2022-09-01 15:12:41
LastEditors: jianzhnie@126.com
LastEditTime: 2022-09-01 15:15:13
Description:

Copyright (c) 2022 by jianzhnie, All Rights Reserved.
'''

import numpy as np
import torch
import torch.nn as nn

from rltoolkit.core.model import Model


def _init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariModel(Model):

    def __init__(self, obs_space, act_space):
        super(AtariModel, self).__init__()

        self.conv1 = _init_layer(nn.Conv2d(4, 32, 8, stride=4))
        self.conv2 = _init_layer(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = _init_layer(nn.Conv2d(64, 64, 3, stride=1))

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = _init_layer(nn.Linear(64 * 7 * 7, 512))

        self.fc_pi = _init_layer(nn.Linear(512, act_space.n), std=0.01)
        self.fc_v = _init_layer(nn.Linear(512, 1), std=1)

    def value(self, obs):
        obs = obs / 255.0
        out = self.relu(self.conv1(obs))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))

        out = self.flatten(out)
        out = self.relu(self.fc(out))
        value = self.fc_v(out)
        return value

    def policy(self, obs):
        obs = obs / 255.0
        out = self.relu(self.conv1(obs))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))

        out = self.flatten(out)
        out = self.relu(self.fc(out))
        logits = self.fc_pi(out)
        return logits
