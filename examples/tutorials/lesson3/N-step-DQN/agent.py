import numpy as np
import torch

from rltoolkit.agent.base_agent import Agent
from rltoolkit.utils.scheduler import LinearDecayScheduler


class Agent(Agent):
    """Agent.

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
        action_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        start_lr (float): initial learning rate
        update_target_step (int): target network update frequency
    """

    def __init__(self,
                 algorithm,
                 action_dim: int,
                 total_step: int,
                 update_target_step: int,
                 start_lr: float = 0.001,
                 end_lr: float = 0.00001,
                 start_epslion: float = 1.0,
                 end_epsilon: float = 0.1,
                 n_step: int = 3,
                 device='cpu'):
        super().__init__(algorithm)
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.action_dim = action_dim
        self.curr_epslion = start_epslion
        self.end_epslion = end_epsilon
        self.end_lr = end_lr
        self.n_step = n_step
        self.device = device
        self.ep_scheduler = LinearDecayScheduler(start_epslion, total_step)
        self.lr_scheduler = LinearDecayScheduler(start_lr, total_step)

    def sample(self, obs) -> int:
        """Sample an action when given an observation, base on the current
        epsilon value, either a greedy action or a random action will be
        returned.

        Args:
            obs: current observation

        Returns:
            act (int): action
        """
        explore = np.random.choice(
            [True, False], p=[self.curr_epslion, 1 - self.curr_epslion])
        if explore:
            act = np.random.randint(self.action_dim)
        else:
            act = self.predict(obs)

        # epslion decay
        self.curr_epslion = max(self.ep_scheduler.step(1), self.end_epslion)
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
        selected_action = self.alg.predict(obs).argmax().item()
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
            self.alg.sync_target()

        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        action = torch.LongTensor(action.reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(reward.reshape(-1, 1)).to(device)
        terminal = torch.FloatTensor(terminal.reshape(-1, 1)).to(device)
        # calculate dqn loss
        loss = self.alg.learn(obs, action, reward, next_obs, terminal,
                              self.n_step)
        self.global_update_step += 1
        # learning rate decay
        for param_group in self.alg.optimizer.param_groups:
            param_group['lr'] = max(self.lr_scheduler.step(1), self.end_lr)
        return loss