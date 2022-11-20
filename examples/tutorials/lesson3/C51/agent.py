import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rltoolkit.models.utils import hard_target_update


class QNetwork(nn.Module):

    def __init__(self,
                 obs_dim: int,
                 hidden_dim: int,
                 act_dim: int,
                 n_atoms: int = 101,
                 v_min: float = -100,
                 v_max: float = 100):
        super().__init__()
        self.n_atoms = n_atoms
        self.register_buffer('atoms',
                             torch.linspace(v_min, v_max, steps=n_atoms))

        self.act_dim = act_dim
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim * n_atoms),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor = None):
        #     x : batch_size * obs_dim
        # logits: batch_size * (act_dim * n_atoms)
        logits = self.network(x)
        # logits: batch_size * act_dim * n_atoms
        logits = logits.view(-1, self.act_dim, self.n_atoms)
        # probability mass function for each action
        # pmfs: batch_size * act_dim * n_atoms
        pmfs = torch.softmax(logits, dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
            action = action.to(dtype=torch.long)
        return action, pmfs[torch.arange(len(x)), action]


class Agent(object):
    """Agent.

    Args:
        action_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        learning_rate (float): initial learning rate
        update_target_step (int): target network update frequency
    """

    def __init__(self,
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 n_atoms: int = 100,
                 v_min: float = -100,
                 v_max: float = 100,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 learning_rate: float = 0.001,
                 update_target_step: int = 100,
                 device='cpu'):
        super().__init__()

        self.gamma = gamma
        self.epsilon = epsilon
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Main network
        self.qnet = QNetwork(
            obs_dim,
            hidden_dim,
            action_dim,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max).to(device)
        # Target network
        self.target_qnet = copy.deepcopy(self.qnet)
        # Create an optimizer
        self.optimizer = Adam(self.qnet.parameters(), lr=learning_rate)

        self.device = device

    def sample(self, obs) -> int:
        """Sample an action when given an observation, base on the current
        epsilon value, either a greedy action or a random action will be
        returned.

        Args:
            obs: current observation

        Returns:
            act (int): action
        """
        # Choose a random action with probability epsilon
        if np.random.rand() <= self.epsilon:
            act = np.random.randint(self.action_dim)
        else:
            # Choose the action with highest Q-value at the current state
            act = self.predict(obs)

        return act

    def predict(self, obs) -> int:
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim) , current observation

        Returns:
            act(int): action
        """
        obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action, pmf = self.qnet(obs)
        action = action.cpu().numpy()
        return action

    def learn(self, obs: torch.Tensor, action: torch.Tensor,
              reward: torch.Tensor, next_obs: torch.Tensor,
              terminal: torch.Tensor) -> float:
        """Update model with an episode data.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim)
            act (np.int32): shape of (batch_size)
            reward (np.float32): shape of (batch_size)
            next_obs (np.float32): shape of (batch_size, obs_dim)
            terminal (np.float32): shape of (batch_size)

        Returns:
            loss (float)
        """
        if self.global_update_step % self.update_target_step == 0:
            hard_target_update(self.qnet, self.target_qnet)

        _, next_pmfs = self.target_qnet(next_obs)
        next_atoms = reward + self.gamma * self.target_qnet.atoms * (1 -
                                                                     terminal)

        # projection
        delta_z = self.target_qnet.atoms[1] - self.target_qnet.atoms[0]
        tz = next_atoms.clamp(self.v_min, self.v_max)

        b = (tz - self.v_min) / delta_z
        l = b.floor().clamp(0, self.n_atoms - 1)
        u = b.ceil().clamp(0, self.n_atoms - 1)

        # (l == u).float() handles the case where bj is exactly an integer
        # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
        d_m_l = (u + (l == u).float() - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = torch.zeros_like(next_pmfs)
        for i in range(target_pmfs.size(0)):
            target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
            target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        action = action.to(dtype=torch.long)
        _, old_pmfs = self.qnet(obs, action)
        loss = torch.mean(
            -(target_pmfs *
              old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1))

        self.optimizer.zero_grad()
        loss.backward()
        # 反向传播更新参数
        self.optimizer.step()
        self.global_update_step += 1
        return loss.item()
