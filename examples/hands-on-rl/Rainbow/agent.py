import copy

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from rltoolkit.utils.scheduler import LinearDecayScheduler
from rltoolkit.utils.utils import check_model_method


class Agent(object):
    """Agent.

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
        act_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        start_lr (float): initial learning rate
        update_target_step (int): target network update frequency
    """

    def __init__(
            self,
            model: torch.nn.Module,
            act_dim: int,
            total_step: int,
            update_target_step: int,
            start_lr: float = 0.001,
            end_lr: float = 0.00001,
            gamma: float = 0.99,
            # Categorical DQN parameters
            batch_size: int = 64,
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 1,
            device: str = 'cpu'):
        super().__init__()
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.act_dim = act_dim
        self.end_lr = end_lr
        self.gamma = gamma
        # Categorical DQN parameters
        self.batch_size = batch_size
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        # N-step Learning
        self.n_step = n_step
        self.device = device
        self.support = torch.linspace(self.v_min, self.v_max,
                                      self.atom_size).to(self.device)
        check_model_method(model, 'forward', self.__class__.__name__)
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.smoothl1_loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=start_lr)
        self.lr_scheduler = LinearDecayScheduler(start_lr, total_step)

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
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        selected_action = self.model(obs).argmax().item()
        return selected_action

    def learn(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_obs: np.ndarray, terminal: np.ndarray) -> float:
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
            self.model.sync_weights_to(self.target_model)

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(
            obs,
            action,
            reward,
            next_obs,
            terminal,
        )
        loss = torch.mean(elementwise_loss)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # calculate dqn loss
        self.global_update_step += 1
        # NoisyNet: reset noise
        self.model.reset_noise()
        self.target_model.reset_noise()
        # learning rate decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.lr_scheduler.step(1), self.end_lr)
        return loss.item()

    def _compute_dqn_loss(self, obs: np.ndarray, action: np.ndarray,
                          reward: np.ndarray, next_obs: np.ndarray,
                          terminal: np.ndarray) -> torch.Tensor:

        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        action = torch.LongTensor(action.reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(reward.reshape(-1, 1)).to(device)
        terminal = torch.FloatTensor(terminal.reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.model(next_obs).argmax(1)
            next_dist = self.target_model.dist(next_obs)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - terminal) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.atom_size,
                               self.batch_size).long().unsqueeze(1).expand(
                                   self.batch_size,
                                   self.atom_size).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                          (next_dist *
                                           (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                          (next_dist *
                                           (b - l.float())).view(-1))

        dist = self.model.dist(obs)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
