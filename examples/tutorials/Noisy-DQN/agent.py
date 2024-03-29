import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from rltoolkit.models.noisynet import NoisyDulingNet, NoisyNet
from rltoolkit.models.utils import hard_target_update


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
                 gamma: float = 0.99,
                 std_init: float = 0.1,
                 v_min: float = 0.0,
                 v_max: float = 200.0,
                 atom_size: int = 51,
                 learning_rate: float = 0.001,
                 update_target_step: int = 100,
                 device: str = 'cpu'):
        super().__init__()

        self.algo = algo
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.action_dim = action_dim
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max,
                                      self.atom_size).to(device)

        # Main network
        if 'duling' in algo:
            self.qnet = NoisyDulingNet(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                atom_size=atom_size,
                std_init=std_init,
                support=self.support)
        else:
            self.qnet = NoisyNet(
                obs_dim, hidden_dim, action_dim, std_init=std_init).to(device)

        # Target network
        self.target_qnet = copy.deepcopy(self.qnet)
        # Create an optimizer
        self.optimizer = Adam(self.qnet.parameters(), lr=learning_rate)
        self.device = device

    def sample(self, obs) -> int:
        """Sample an action when given an observation,
        NoisyNet: no epsilon greedy action selection

        Args:
            obs: current observation

        Returns:
            act (int): action
        """
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
        if obs.ndim == 1:  # if obs is 1 dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action = self.qnet(obs).argmax().item()
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

        action = action.to(self.device, dtype=torch.long)

        # Prediction Q(s)
        pred_value = self.qnet(obs).gather(1, action)

        # Target for Q regression
        if self.algo in ['noisy_dqn', 'noisy_duling_dqn']:
            next_q_value = self.target_qnet(next_obs).max(1, keepdim=True)[0]

        elif self.algo in ['noisy_ddqn', 'noisy_duling_ddqn']:
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
        # NoisyNet: reset noise
        self.qnet.reset_noise()
        self.target_qnet.reset_noise()
        return loss.item()
