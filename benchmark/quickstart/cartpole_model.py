'''
Author: jianzhnie
Date: 2022-09-02 12:19:54
LastEditors: jianzhnie
LastEditTime: 2022-09-02 19:47:59
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from rltoolkit.core.models.base_model import Model


class CartpoleModel(Model):
    """Linear network to solve Cartpole problem.

    Args:
        obs_dim (int): Dimension of observation space.
        act_dim (int): Dimension of action space.
    """

    def __init__(self, obs_dim, act_dim):
        super(CartpoleModel, self).__init__()
        hid1_size = act_dim * 10
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, act_dim)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = self.fc2(out)
        prob = F.softmax(out, dim=-1)
        return prob
