'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import collections
import random
from typing import Dict

import numpy as np
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer(object):
    """A simple FIFO experience replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs_buf = np.zeros(
            combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(
            combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(
            combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float,
              next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size, replace=False)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs])
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in batch.items()
        }

    def __len__(self) -> int:
        return self.size


class ReplayBuffer_:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
