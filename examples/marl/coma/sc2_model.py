'''
Author: jianzhnie
Date: 2022-09-02 19:35:02
LastEditors: jianzhnie
LastEditTime: 2022-09-03 10:52:35
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from rltoolkit.models.base_model import Model


class ComaModel(Model):

    def __init__(self, config):
        super(ComaModel, self).__init__()
        self.n_actions = config['n_actions']
        self.n_agents = config['n_agents']
        self.state_shape = config['state_shape']
        self.obs_shape = config['obs_shape']

        actor_input_dim = self._get_actor_input_dim()
        critic_input_dim = self._get_critic_input_dim()

        self.actor_model = ActorModel(actor_input_dim, self.n_actions)
        self.critic_model = CriticModel(critic_input_dim, self.n_actions)

    def policy(self, obs, hidden_state):
        return self.actor_model(obs, hidden_state)

    def value(self, inputs):
        return self.critic_model(inputs)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

    def _get_actor_input_dim(self):
        input_shape = self.obs_shape  # obs: 30 in 3m map
        input_shape += self.n_actions  # agent's last action (one_hot): 9 in 3m map
        input_shape += self.n_agents  # agent's one_hot id: 3 in 3m map
        return input_shape  # 30 + 9 + 3 = 42

    def _get_critic_input_dim(self):
        input_shape = self.state_shape  # state: 48 in 3m map
        input_shape += self.obs_shape  # obs: 30 in 3m map
        input_shape += self.n_agents  # agent_id: 3 in 3m map
        input_shape += self.n_actions * self.n_agents * 2  # all agents' action and last_action (one-hot): 54 in 3m map
        return input_shape  # 48 + 30+ 3 = 135


# all agents share one actor network
class ActorModel(nn.Module):
    """input: obs, include the agent's id and last action, shape: (batch, obs_shape + n_action + n_agents).
       output: one agent's q(obs, act)
    """

    def __init__(self,
                 input_shape: int = None,
                 n_actions: int = None,
                 rnn_hidden_dim: int = 64):
        super(ActorModel, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # new hidden states
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self,
                obs: torch.Tensor = None,
                hidden_state: torch.Tensor = None):

        x = F.relu(self.fc1(obs), inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class CriticModel(nn.Module):
    """ inputs: [ s(t), o(t)_a, u(t)_a, agent_a, u(t-1) ], shape: (Batch, input_shape)
        output: Q,   shape: (Batch, n_actions)
        Batch = ep_num * n_agents
    """

    def __init__(self,
                 input_shape: int = None,
                 n_actions: int = None,
                 hidden_size: int = 128):

        super(CriticModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, inputs):
        hid1 = F.relu(self.fc1(inputs), inplace=True)
        hid2 = F.relu(self.fc2(hid1), inplace=True)
        Q = self.fc3(hid2)
        return Q
