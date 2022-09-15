'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import random
from typing import Dict, List, Tuple, Union

import numpy as np

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

    def __init__(self, obs_dim: Union[int, Tuple], act_dim: Union[int, Tuple],
                 max_size: int):
        self.obs = np.zeros(
            combined_shape(max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros(
            combined_shape(max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros(
            combined_shape(max_size, act_dim), dtype=np.float32)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.terminal = np.zeros(max_size, dtype=np.float32)

        self._curr_ptr = 0
        self._curr_size = 0
        self.max_size = max_size

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

        self.obs[self._curr_ptr] = obs
        self.next_obs[self._curr_ptr] = next_obs
        self.action[self._curr_ptr] = act
        self.reward[self._curr_ptr] = rew
        self.terminal[self._curr_ptr] = terminal

        self._curr_ptr = (self._curr_ptr + 1) % self.max_size
        self._curr_size = min(self._curr_size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """sample a batch from replay memory.

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        idxs = np.random.randint(self._curr_size, size=batch_size)

        batch = dict(
            obs=self.obs[idxs],
            next_obs=self.next_obs[idxs],
            action=self.action[idxs],
            reward=self.reward[idxs],
            terminal=self.terminal[idxs],
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
            obs=self.obs,
            action=self.action,
            reward=self.reward,
            terminal=self.terminal,
            next_obs=self.next_obs,
            other=other)

    def load(self, pathname: str) -> None:
        """load replay memory from local file (numpy file format: *.npz)."""
        data = np.load(pathname)
        other = data['other']
        if int(other[0]) > self.max_size:
            logger.warn('loading from a bigger size rpm!')
        self._curr_size = min(int(other[0]), self.max_size)
        self._curr_ptr = min(int(other[1]), self.max_size - 1)

        self.obs[:self._curr_size] = data['obs'][:self._curr_size]
        self.action[:self._curr_size] = data['action'][:self._curr_size]
        self.reward[:self._curr_size] = data['reward'][:self._curr_size]
        self.terminal[:self._curr_size] = data['terminal'][:self._curr_size]
        self.next_obs[:self._curr_size] = data['next_obs'][:self._curr_size]
        logger.info('[load rpm]memory loade from {}'.format(pathname))


class PrioritizedReplayBuffer(ReplayBuffer):
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
                 act_dim: int,
                 size: int,
                 alpha: float = 0.6):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_dim, act_dim, size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, obs: np.ndarray, act: int, rew: float,
              next_obs: np.ndarray, terminal: bool):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, terminal)

        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self,
                     beta: float = 0.4,
                     batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        terminal = self.terminal_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
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
