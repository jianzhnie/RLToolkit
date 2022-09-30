from typing import Dict, List, Tuple, Union

import numpy as np
import torch


class ReplayBuffer(object):

    def __init__(
        self,
        obs_dim: Union[int, Tuple],
        num_agents: int,
        max_size: int,
        batch_size: int,
    ):

        self.obs_buf = np.zeros((max_size, num_agents, obs_dim),
                                dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, num_agents, obs_dim),
                                     dtype=np.float32)
        self.action_buf = np.zeros((max_size, num_agents), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, num_agents), dtype=np.float32)
        self.terminal_buf = np.zeros((max_size, num_agents), dtype=np.float32)

        self._curr_ptr = 0
        self._curr_size = 0
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.max_size = max_size
        self.batch_size = batch_size

    def store(self, obs_all: List, act_all: List, reward_all: List,
              next_obs_all: List, terminal_all: List):
        agent_idx = 0
        for transition in zip(obs_all, act_all, reward_all, next_obs_all,
                              terminal_all):
            obs, act, reward, next_obs, terminal = transition

            self.obs_buf[self._curr_ptr, agent_idx] = obs
            self.next_obs_buf[self._curr_ptr, agent_idx] = next_obs
            self.action_buf[self._curr_ptr, agent_idx] = act
            self.reward_buf[self._curr_ptr, agent_idx] = reward
            self.terminal_buf[self._curr_ptr, agent_idx] = terminal

            agent_idx += 1

        self._curr_ptr = (self._curr_ptr + 1) % self.max_size
        self._curr_size = min(self._curr_size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(self._curr_size, size=self.batch_size)

        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.reward_buf[idxs],
            terminal=self.terminal_buf[idxs],
            indices=idxs,  # for N -step Learning
        )

        return batch

    def sample_chunk(self, chunk_size) -> Dict[str, np.ndarray]:
        start_idx = np.random.randint(
            self._curr_size - chunk_size, size=self.batch_size)

        obs_chunk = torch.tensor(self.obs_buf[idx:idx + chunk_size]
                                 for idx in start_idx).view(
                                     self.batch_size, chunk_size,
                                     self.num_agents, self.obs_dim)

        next_obs_chunk = torch.tensor(self.next_obs_buf[idx:idx + chunk_size]
                                      for idx in start_idx).view(
                                          self.batch_size, chunk_size,
                                          self.num_agents, self.obs_dim)

        action_chunk = torch.tensor(self.action_buf[idx:idx + chunk_size]
                                    for idx in start_idx).view(
                                        self.batch_size, chunk_size,
                                        self.num_agents, self.obs_dim)

        reward_chunk = torch.tensor(self.reward_buf[idx:idx + chunk_size]
                                    for idx in start_idx).view(
                                        self.batch_size, chunk_size,
                                        self.num_agents, self.obs_dim)

        terminal_chunk = torch.tensor(self.terminal_buf[idx:idx + chunk_size]
                                      for idx in start_idx).view(
                                          self.batch_size, chunk_size,
                                          self.num_agents, self.obs_dim)

        batch = dict(
            obs=obs_chunk,
            next_obs=next_obs_chunk,
            action=action_chunk,
            reward=reward_chunk,
            terminal=terminal_chunk)

        return batch

    def size(self) -> int:
        """get current size of replay memory."""
        return self._curr_size

    def __len__(self):
        return self._curr_size
