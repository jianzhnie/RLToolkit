'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixerModel(nn.Module):
    '''
    input: n_agents' agent_qs (a scalar for each agent)
    output: a scalar (Q)
    '''

    def __init__(self,
                 n_agents: int = None,
                 state_shape: int = None,
                 mixing_embed_dim: int = 32,
                 hypernet_layers: int = 2,
                 hypernet_embed_dim: int = 64):
        super(QMixerModel, self).__init__()

        self.n_agents = n_agents
        self.state_shape = state_shape
        self.embed_dim = mixing_embed_dim
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_shape,
                                       self.embed_dim * self.n_agents)
            self.hyper_w_2 = nn.Linear(self.state_shape, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_shape, hypernet_embed_dim), nn.ReLU(),
                nn.Linear(hypernet_embed_dim, self.embed_dim * self.n_agents))
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_shape, hypernet_embed_dim), nn.ReLU(),
                nn.Linear(hypernet_embed_dim, self.embed_dim))
        else:
            raise ValueError('hypernet_layers should be "1" or "2"!')

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_shape, self.embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.state_shape, self.embed_dim), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        '''
        Args:
            agent_qs (torch.Tensor): (batch_size, T, n_agents)
            states (torch.Tensor):   (batch_size, T, state_shape)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
        '''
        batch_size = agent_qs.size(0)
        # states : (batch_size * T, state_shape)
        states = states.reshape(-1, self.state_shape)
        # agent_qs: (batch_size * T, 1, n_agents)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer w and b
        w1 = torch.abs(self.hyper_w_1(states))
        # w1: (batch_size * T, n_agents, embed_dim)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)

        # Second layer w and b
        # w2 : (batch_size * T, embed_dim)
        w2 = torch.abs(self.hyper_w_2(states))
        # w2 : (batch_size * T, embed_dim, 1)
        w2 = w2.view(-1, self.embed_dim, 1)
        # State-dependent bias
        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        # First hidden layer
        # hidden: (batch_size * T,  1, embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Compute final output
        # y: (batch_size * T,  1, 1)
        y = torch.bmm(hidden, w2) + b2
        # Reshape and return
        # q_total: (batch_size, T, 1)
        q_total = y.view(batch_size, -1, 1)
        return q_total
