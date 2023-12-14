'''
Author: jianzhnie
Date: 2022-09-02 14:36:16
LastEditors: jianzhnie
LastEditTime: 2022-09-03 15:52:00
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from rltoolkit.models.base_model import Model


class ActorCritic(Model):

    def __init__(self, act_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=8,
                               stride=4,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=2)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=0)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 9 * 9, 512)

        self.fc_pi = nn.Linear(512, act_dim)
        self.fc_v = nn.Linear(512, 1)

        # params init
        self._init_parameters()

    def policy(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)
        x = F.relu(self.fc(x))

        logits = self.fc_pi(x)
        return logits

    def value(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)
        x = F.relu(self.fc(x))
        values = self.fc_v(x)
        values = torch.squeeze(values, dim=1)

        return values

    def policy_and_value(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)
        x = F.relu(self.fc(x))

        values = self.fc_v(x)
        logits = self.fc_pi(x)
        values = torch.squeeze(values, dim=1)
        return logits, values

    # In pytorch, most layers/convs are initialized with kaiming_normal_ method, which may perform worse in the pong game.
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # xavier init
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
