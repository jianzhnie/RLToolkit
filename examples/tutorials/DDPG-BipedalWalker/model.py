import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize parameters and build model.

        Params:
            obs_dim (int): Dimension of each obs
            action_dim (int): Dimension of each action
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Build an actor (policy) network that maps obss -> actions."""
        x = self.fc1(obs)
        x = self.relu(x)
        x = self.fc2(x)
        out = torch.tanh(x)
        return out


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, obs_dim, action_dim, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.

        Params:
            obs_dim (int): Dimension of each obs
            action_dim (int): Dimension of each action
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, action):
        """Build a critic (value) network that maps (obs, action) pairs ->
        Q-values."""
        cat = torch.cat((obs, action), dim=1)
        out = F.leaky_relu(self.fc1(cat))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        return out
