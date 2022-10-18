'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import random
from collections import deque
from typing import Any, Deque, Dict, List, Tuple, Union

import numpy as np
import torch

from rltoolkit.utils import logger

from ..utils.segment_tree import MinSegmentTree, SumSegmentTree


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer(object):
    """A simple FIFO experience replay buffer for off-policy RL or offline RL.

    Args:
        max_size (int): max size of replay memory
        obs_dim (int or tuple): observation shape
        act_dim (int or tuple): action shape
    """

    def __init__(self,
                 max_size: int,
                 obs_dim: Union[int, Tuple],
                 action_dim: Union[int, Tuple] = 1,
                 batch_size: int = 32):
        self.obs_buf = np.zeros(
            combined_shape(max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(
            combined_shape(max_size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros(
            combined_shape(max_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.terminal_buf = np.zeros((max_size, 1), dtype=np.float32)

        self._curr_ptr = 0
        self._curr_size = 0
        self.max_size = max_size
        self.batch_size = batch_size

    def append(self, obs: np.ndarray, act: np.ndarray, rew: float,
               next_obs: np.ndarray, terminal: bool) -> None:
        """add an experience sample at the end of replay memory.

        Args:
            obs (float32): observation, shape of obs_dim
            act (int32 in Continuous control environment, float32 in Continuous control environment): action, shape of act_dim
            reward (float32): reward
            next_obs (float32): next observation, shape of obs_dim
            terminal (bool): terminal of an episode or not
        """

        self.obs_buf[self._curr_ptr] = obs
        self.next_obs_buf[self._curr_ptr] = next_obs
        self.action_buf[self._curr_ptr] = act
        self.reward_buf[self._curr_ptr] = rew
        self.terminal_buf[self._curr_ptr] = terminal

        self._curr_ptr = (self._curr_ptr + 1) % self.max_size
        self._curr_size = min(self._curr_size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """sample a batch from replay memory.

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
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

    def size(self) -> int:
        """get current size of replay memory."""
        return self._curr_size

    def __len__(self):
        return self._curr_size

    def save(self, pathname: str) -> None:
        """save replay memory to local file (numpy file format: *.npz)."""
        other = np.array([self._curr_size, self._curr_ptr], dtype=np.int32)
        np.savez(
            pathname,
            obs=self.obs_buf,
            action=self.action_buf,
            reward=self.reward_buf,
            terminal=self.terminal_buf,
            next_obs=self.next_obs_buf,
            other=other)

    def load(self, pathname: str) -> None:
        """load replay memory from local file (numpy file format: *.npz)."""
        data = np.load(pathname)
        other = data['other']
        if int(other[0]) > self.max_size:
            logger.warn('loading from a bigger size rpm!')
        self._curr_size = min(int(other[0]), self.max_size)
        self._curr_ptr = min(int(other[1]), self.max_size - 1)

        self.obs_buf[:self._curr_size] = data['obs'][:self._curr_size]
        self.action_buf[:self._curr_size] = data['action'][:self._curr_size]
        self.reward_buf[:self._curr_size] = data['reward'][:self._curr_size]
        self.terminal_buf[:self._curr_size] = data['terminal'][:self.
                                                               _curr_size]
        self.next_obs_buf[:self._curr_size] = data['next_obs'][:self.
                                                               _curr_size]
        logger.info('[load rpm]memory loade from {}'.format(pathname))


class TorchReplayBuffer(object):
    """A simple FIFO experience replay buffer for off-policy RL or offline RL.

    Args:
        max_size (int): max size of replay memory
        obs_dim (int or tuple): observation shape
        act_dim (int or tuple): action shape
    """

    def __init__(self,
                 obs_dim: Union[int, Tuple],
                 max_size: int,
                 batch_size: int,
                 device: Any = None):
        self.obs_buf = torch.empty(
            combined_shape(max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = torch.empty(
            combined_shape(max_size, obs_dim), dtype=np.float32)
        self.action_buf = torch.empty(max_size, dtype=np.float32)
        self.reward_buf = torch.empty(max_size, dtype=np.float32)
        self.terminal_buf = torch.empty(max_size, dtype=np.float32)
        self.device = device

        self._curr_ptr = 0
        self._curr_size = 0
        self.max_size = max_size
        self.batch_size = batch_size

    def append(self, obs: np.ndarray, act: np.ndarray, reward: float,
               next_obs: np.ndarray, terminal: bool) -> None:
        """Store the experience to self.memory (replay buffer)

        Args:
            obs (float32): observation, shape of obs_dim
            act (int32 in Continuous control environment, float32 in Continuous control environment): action, shape of act_dim
            reward (float32): reward
            next_obs (float32): next observation, shape of obs_dim
            terminal (bool): terminal of an episode or not
        """
        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.LongTensor([act]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        terminal = torch.BoolTensor([terminal]).to(self.device)

        self.obs_buf[self._curr_ptr] = obs
        self.next_obs_buf[self._curr_ptr] = next_obs
        self.action_buf[self._curr_ptr] = action
        self.reward_buf[self._curr_ptr] = reward
        self.terminal_buf[self._curr_ptr] = terminal

        self._curr_ptr = (self._curr_ptr + 1) % self.max_size
        self._curr_size = min(self._curr_size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """sample a batch from replay memory.

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """

        idxs = torch.randint(
            self._curr_size, size=self.batch_size, device=self.device)

        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.reward_buf[idxs],
            terminal=self.terminal_buf[idxs],
            indices=idxs,  # for N -step Learning
        )

        return batch

    def size(self) -> int:
        """get current size of replay memory."""
        return self._curr_size

    def __len__(self):
        return self._curr_size


class MultiStepReplayBuffer(ReplayBuffer):
    """A simple numpy replay buffer."""

    def __init__(
        self,
        obs_dim: int,
        max_size: int,
        batch_size: int = 32,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        super(MultiStepReplayBuffer, self).__init__(obs_dim, max_size,
                                                    batch_size)

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def append(
        self, obs: np.ndarray, act: np.ndarray, rew: float,
        next_obs: np.ndarray, terminal: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:

        transition = (obs, act, rew, next_obs, terminal)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return

        # make a n-step transition
        # firstly, get curr obs and curr action
        obs, act = self.n_step_buffer[0][:2]
        # get the next_obs, reward, terminal in n_step_buffer deque
        rew, next_obs, terminal = self._get_n_step_info(
            self.n_step_buffer, self.gamma)

        self.obs_buf[self._curr_ptr] = obs
        self.next_obs_buf[self._curr_ptr] = next_obs
        self.action_buf[self._curr_ptr] = act
        self.reward_buf[self._curr_ptr] = rew
        self.terminal_buf[self._curr_ptr] = terminal

        self._curr_ptr = (self._curr_ptr + 1) % self.max_size
        self._curr_size = min(self._curr_size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch_from_idxs(self,
                               indices: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        batch = dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            action=self.action_buf[indices],
            reward=self.reward_buf[indices],
            terminal=self.terminal_buf[indices])
        return batch

    def _get_n_step_info(self, n_step_buffer: Deque,
                         gamma: float) -> Tuple[float, np.array, bool]:
        """Return n step rew, next_obs, and terminal."""
        # info of the last transition
        rew, next_obs, terminal = n_step_buffer[-1][-3:]

        # info of the n-1 transition
        sub_n_step_buffer = list(n_step_buffer)[:-1]
        for transition in reversed(sub_n_step_buffer):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, terminal = (n_o, d) if d else (next_obs, terminal)

        return rew, next_obs, terminal


class PrioritizedReplayBuffer(MultiStepReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
    """

    def __init__(self,
                 obs_dim: int,
                 max_size: int,
                 batch_size: int = 32,
                 n_step: int = 1,
                 gamma: float = 0.99,
                 alpha: float = 0.6):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer,
              self).__init__(obs_dim, max_size, batch_size, n_step, gamma)
        # for Prioritized Replay buffer
        self.max_priority = 1.0
        self.tree_ptr = 0

        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def apeend(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        terminal: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().append(obs, act, rew, next_obs, terminal)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        action = self.action_buf[indices]
        reward = self.reward_buf[indices]
        terminal = self.terminal_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            terminal=terminal,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self))**(-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self))**(-beta)
        weight = weight / max_weight

        return weight
