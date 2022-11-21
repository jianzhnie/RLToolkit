'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import random
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from stable_baselines3.common.vec_env import VecNormalize

from rltoolkit.env.preprocessing import get_action_dim, get_obs_shape
from rltoolkit.utils import logger
from rltoolkit.utils.name_dict import ReplayBufferSamples, RolloutBufferSamples

from ..utils.segment_tree import MinSegmentTree, SumSegmentTree

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class BaseBuffer(ABC):
    """Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = 'cpu',
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.curr_ptr = 0
        self.curr_size = 0
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """Swap and then flatten axes 0 (buffer_size) and 1 (n_envs) to convert
        shape from [n_steps, n_envs, ...] (when ... is the shape of the
        features)

        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1, )
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        return self.curr_size

    def add(self, *args, **kwargs) -> None:
        """Add elements to the buffer."""
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """Add a new batch of transitions to the buffer."""
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """Reset the buffer."""
        self.curr_ptr = 0
        self.curr_size = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        batch_inds = np.random.randint(self.curr_size, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        device: Union[torch.device, str] = 'cpu',
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):

        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                'ReplayBuffer does not support optimize_memory_usage = True '
                'and handle_timeout_termination = True simultaneously.')
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs),
                                dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs),
                              dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs),
                                 dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    'This system does not have apparently enough memory to store the complete '
                    f'replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB'
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, ) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs, ) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.curr_ptr] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.curr_ptr + 1) %
                              self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.curr_ptr] = np.array(next_obs).copy()

        self.actions[self.curr_ptr] = np.array(action).copy()
        self.rewards[self.curr_ptr] = np.array(reward).copy()
        self.dones[self.curr_ptr] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.curr_ptr] = np.array(
                [info.get('TimeLimit.truncated', False) for info in infos])

        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def sample(self,
               batch_size: int,
               env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """Sample elements from the replay buffer. Custom sampling when using
        memory efficient variant, as we should not sample the element with
        index `self.pos` See https://github.com/DLR-RM/stable-
        baselines3/pull/28#issuecomment-637559274.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.size == self.buffer_size:
            batch_inds = np.random.randint(1, self.curr_size, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.curr_size, size=batch_size)

        return self._get_samples(batch_inds, env=env)

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(
            0, high=self.n_envs, size=(len(batch_inds), ))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size,
                                  env_indices, :], env)
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :],
                                env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] *
             (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class SimpleReplayBuffer(object):
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
                 batch_size: int = 32,
                 device: str = 'cpu'):
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
        self.device = device

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

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)

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
        batch = {
            key: torch.tensor(val, device=self.device)
            for (key, val) in batch.items()
        }

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
    """Torch Version.

    A simple FIFO experience replay buffer for off-policy RL or offline RL.


    Args:
        max_size (int): max size of replay memory
        obs_dim (int or tuple): observation shape
        act_dim (int or tuple): action shape
    """

    def __init__(self,
                 max_size: int,
                 obs_dim: Union[int, Tuple],
                 action_dim: Union[int, Tuple] = 1,
                 batch_size: int = 32,
                 device: str = 'cpu'):
        self.obs_buf = torch.empty(
            combined_shape(max_size, obs_dim),
            dtype=torch.float32,
            device=device)
        self.next_obs_buf = torch.empty(
            combined_shape(max_size, obs_dim),
            dtype=torch.float32,
            device=device)
        self.action_buf = torch.empty(
            combined_shape(max_size, action_dim),
            dtype=torch.float32,
            device=device)
        self.reward_buf = torch.empty((max_size, 1),
                                      dtype=torch.float32,
                                      device=device)
        self.terminal_buf = torch.empty((max_size, 1),
                                        dtype=torch.float32,
                                        device=device)
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
        action = torch.FloatTensor(np.array([act])).to(self.device)
        reward = torch.FloatTensor(np.array([reward])).to(self.device)
        terminal = torch.BoolTensor(np.array([terminal])).to(self.device)

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
            self._curr_size, size=(self.batch_size, ), device=self.device)

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


class MultiStepReplayBuffer(SimpleReplayBuffer):
    """A simple numpy replay buffer."""

    def __init__(self,
                 max_size: int,
                 obs_dim: Union[int, Tuple],
                 action_dim: Union[int, Tuple] = 1,
                 batch_size: int = 32,
                 n_step: int = 3,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        super(MultiStepReplayBuffer,
              self).__init__(max_size, obs_dim, action_dim, batch_size, device)

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
