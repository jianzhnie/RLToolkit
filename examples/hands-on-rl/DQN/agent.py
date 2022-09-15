from typing import Dict

import numpy as np
import torch

from rltoolkit.agent.base_agent import Agent
from rltoolkit.utils.scheduler import LinearDecayScheduler


class Agent(Agent):
    """Agent.

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
        act_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        start_lr (float): initial learning rate
        update_target_step (int): target network update frequency
    """

    def __init__(self,
                 algorithm,
                 act_dim: int,
                 start_lr: float,
                 start_epslion: float,
                 total_step: int,
                 update_target_step: int,
                 device='cpu'):
        super().__init__(algorithm)
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.act_dim = act_dim
        self.curr_ep = start_epslion
        self.ep_end = 0.1
        self.lr_end = 0.00001
        self.device = device
        self.ep_scheduler = LinearDecayScheduler(1, total_step)
        self.lr_scheduler = LinearDecayScheduler(start_lr, total_step)

    def sample(self, obs):
        """Sample an action when given an observation, base on the current
        epsilon value, either a greedy action or a random action will be
        returned.

        Args:
            obs: current observation

        Returns:
            act (int): action
        """
        explore = np.random.choice([True, False],
                                   p=[self.curr_ep, 1 - self.curr_ep])
        if explore:
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)

        # epslion decay
        self.curr_ep = max(self.ep_scheduler.step(1), self.ep_end)
        return act

    def predict(self, obs):
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim) , current observation

        Returns:
            act(int): action
        """
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        selected_action = self.alg.predict(obs).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def learn(self, samples):
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
            self.alg.sync_target()

        self.global_update_step += 1
        loss = self._compute_dqn_loss(samples=samples)
        # learning rate decay
        for param_group in self.alg.optimizer.param_groups:
            param_group['lr'] = max(self.lr_scheduler.step(1), self.lr_end)
        return loss

    def _compute_dqn_loss(self, samples: Dict[str,
                                              np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(samples['obs']).to(device)
        next_obs = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.LongTensor(samples['action'].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples['reward'].reshape(-1, 1)).to(device)
        terminal = torch.FloatTensor(samples['terminal'].reshape(-1,
                                                                 1)).to(device)
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        # calculate dqn loss
        loss = self.alg.learn(obs, action, reward, next_obs, terminal)
        return loss
