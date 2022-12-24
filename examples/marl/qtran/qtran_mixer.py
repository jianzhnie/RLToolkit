'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import torch
import torch.nn as nn


class QTransModel(nn.Module):

    def __init__(self,
                 n_agents: int = None,
                 n_actions: int = None,
                 state_dim: int = None,
                 rnn_hidden_dim: int = 32,
                 mixing_embed_dim: int = 32,
                 qtran_arch: str = 'qtran_paper',
                 network_size: str = 'small'):
        super(QTransModel, self).__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.qtran_arch = qtran_arch
        self.network_size = network_size
        ae_input = self.rnn_hidden_dim + self.n_actions

        # Q(s,u)
        if self.qtran_arch == 'coma_critic':
            # Q takes [state, u] as input
            q_input_size = self.state_dim + (self.n_agents * self.n_actions)
        elif self.qtran_arch == 'qtran_paper':
            # Q takes [state, agent_action_observation_encodings]
            q_input_size = self.state_dim + self.rnn_hidden_dim + self.n_actions
        else:
            raise Exception('{} is not a valid QTran architecture'.format(
                self.qtran_arch))

        if self.network_size == 'small':
            self.Q = nn.Sequential(
                nn.Linear(q_input_size, self.mixing_embed_dim), nn.ReLU(),
                nn.Linear(self.mixing_embed_dim, self.mixing_embed_dim),
                nn.ReLU(), nn.Linear(self.mixing_embed_dim, 1))

            # V(s)
            self.V = nn.Sequential(
                nn.Linear(self.state_dim, self.mixing_embed_dim), nn.ReLU(),
                nn.Linear(self.mixing_embed_dim, self.mixing_embed_dim),
                nn.ReLU(), nn.Linear(self.mixing_embed_dim, 1))

            self.action_encoding = nn.Sequential(
                nn.Linear(ae_input, ae_input), nn.ReLU(),
                nn.Linear(ae_input, ae_input))

        elif self.network_size == 'big':
            self.Q = nn.Sequential(
                nn.Linear(q_input_size, self.mixing_embed_dim), nn.ReLU(),
                nn.Linear(self.mixing_embed_dim, self.mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(self.mixing_embed_dim, self.mixing_embed_dim),
                nn.ReLU(), nn.Linear(self.mixing_embed_dim, 1))
            # V(s)
            self.V = nn.Sequential(
                nn.Linear(self.state_dim, self.mixing_embed_dim), nn.ReLU(),
                nn.Linear(self.mixing_embed_dim, self.mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(self.mixing_embed_dim, self.mixing_embed_dim),
                nn.ReLU(), nn.Linear(self.mixing_embed_dim, 1))
            self.action_encoding = nn.Sequential(
                nn.Linear(ae_input, ae_input), nn.ReLU(),
                nn.Linear(ae_input, ae_input))
        else:
            raise Exception('{} is not a valid QTran architecture'.format(
                self.network_size))

    def forward(self, states, hidden_states, actions=None):
        '''
        Args:
            agent_qs (torch.Tensor): (batch_size, T, n_agents)
            states (torch.Tensor):   (batch_size, T, state_shape)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
        '''
        batch_size = states.size(0)
        episode_len = states.size(1)

        # states : (batch_size * T, state_shape)
        states = states.reshape(-1, self.state_dim)

        if self.qtran_arch == 'coma_critic':
            # It will arrive as (bs, ts, agents, actions), we need to reshape it
            actions = actions.reshape(-1, self.n_agents * self.n_actions)

            inputs = torch.cat([states, actions], dim=1)

        elif self.qtran_arch == 'qtran_paper':
            # It will arrive as (bs, ts, agents, actions), we need to reshape it
            actions = actions.reshape(batch_size * episode_len, self.n_agents,
                                      self.n_actions)

            hidden_states = hidden_states.reshape(batch_size * episode_len,
                                                  self.n_agents, -1)
            state_action = torch.cat([hidden_states, actions], dim=2)

            state_action_reshape = state_action.reshape(
                batch_size * episode_len, self.n_agents, -1)

            state_action_encoding = self.action_encoding(state_action_reshape)

            state_action_encoding = state_action_encoding.reshape(
                batch_size * episode_len, self.n_agents, -1)
            # Sum across agents
            state_action_encoding = state_action_encoding.sum(dim=1)
            inputs = torch.cat([states, state_action_encoding], dim=1)

        q_outputs = self.Q(inputs)
        v_outputs = self.V(states)

        return q_outputs, v_outputs
