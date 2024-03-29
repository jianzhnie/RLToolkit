import copy

import numpy as np
import torch
import torch.nn.functional as F
from network import DulingNet, QNet
from torch.optim import Adam

from rltoolkit.models.utils import hard_target_update
from rltoolkit.utils import LinearDecayScheduler


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
                 algo: str = 'dqn',
                 total_steps: int = 10000,
                 gamma: float = 0.99,
                 exploration_start: float = 1.0,
                 min_exploration: float = 0.01,
                 learning_rate: float = 0.001,
                 update_target_step: int = 100,
                 device='cpu'):
        super().__init__()

        self.algo = algo
        self.gamma = gamma
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.learning_rate = learning_rate
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.action_dim = action_dim

        # Main network
        if 'duling' in algo:
            self.qnet = DulingNet(obs_dim, hidden_dim, action_dim).to(device)
        else:
            self.qnet = QNet(obs_dim, hidden_dim, action_dim).to(device)
        # Target network
        self.target_qnet = copy.deepcopy(self.qnet)
        # Create an optimizer
        self.optimizer = Adam(self.qnet.parameters(), lr=learning_rate)
        self.ep_scheduler = LinearDecayScheduler(exploration_start,
                                                 total_steps * 0.8)

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
        if np.random.rand() <= self.exploration:
            act = np.random.randint(self.action_dim)
        else:
            # Choose the action with highest Q-value at the current state
            act = self.predict(obs)

        # update exploration
        self.exploration = max(self.ep_scheduler.step(), self.min_exploration)

        return act

    def predict(self, obs) -> int:
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim) , current observation

        Returns:
            act(int): action
        """
        if obs.ndim == 1:  # if obs is 1 dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        selected_action = self.qnet(obs).argmax().item()
        return selected_action

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

        action = action.to(self.device, dtype=torch.long)
        # Prediction Q(s)
        pred_value = self.qnet(obs).gather(1, action)

        # Target for Q regression
        if self.algo in ['dqn', 'duling_dqn']:
            next_q_value = self.target_qnet(next_obs).max(1, keepdim=True)[0]

        elif self.algo in ['ddqn', 'duling_ddqn']:
            greedy_action = self.qnet(next_obs).max(dim=1, keepdim=True)[1]
            next_q_value = self.target_qnet(next_obs).gather(1, greedy_action)

        target = reward + (1 - terminal) * self.gamma * next_q_value

        # TD误差目标
        loss = F.mse_loss(pred_value, target)
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        loss.backward()
        # 反向传播更新参数
        self.optimizer.step()
        self.global_update_step += 1
        return loss.item()
