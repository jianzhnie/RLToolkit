'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self,
                 input_shape: int = None,
                 n_actions: int = None,
                 rnn_hidden_dim: int = 64):
        super(RNNModel, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self,
                obs: torch.Tensor = None,
                hidden_state: torch.Tensor = None):
        x = F.relu(self.fc1(obs), inplace=True)
        if hidden_state is not None:
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)

        h = self.rnn(x, h_in)
        q = self.fc2(h)  # (batch_size, n_actions)
        return q, h


class VDNMixer(nn.Module):
    """Computes total Q values given agent q values and global states."""

    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, q_values, states=None):
        return torch.sum(q_values, dim=2, keepdim=True)
