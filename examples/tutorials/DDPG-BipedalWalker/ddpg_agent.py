import copy
import random
import sys
from collections import deque, namedtuple
from typing import Any, Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
from model import Actor, Critic
from torch.optim import Adam

sys.path.append('../../../../')
from rltoolkit.models.ounoise import OUNoise


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_dim, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_dim = action_dim
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple(
            'Experience',
            field_names=['obs', 'action', 'reward', 'next_obs', 'done'])

    def append(self, obs, action, reward, next_obs, done):
        """Add a new experience to memory."""
        e = self.experience(obs, action, reward, next_obs, done)
        self.memory.append(e)

    def sample_batch(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        device = self.device

        obs = torch.from_numpy(
            np.vstack([e.obs for e in experiences
                       if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(device)
        next_obs = torch.from_numpy(
            np.vstack([e.next_obs for e in experiences
                       if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences
                       if e is not None]).astype(np.uint8)).float().to(device)

        batch = dict(
            obs=obs,
            next_obs=next_obs,
            action=actions,
            reward=rewards,
            terminal=dones,
        )

        return batch

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def size(self) -> int:
        """get current size of replay memory."""
        return len(self.memory)


class Agent(object):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 env: gym.Env,
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 weight_decay: float,
                 action_bound: float,
                 initial_random_steps: int,
                 tau: float,
                 gamma: float,
                 device: Any = None):
        """Initialize an Agent object.

        Params
        ======
            obs_dim (int): dimension of each obs
            action_dim (int): dimension of each action
            random_seed (int): random seed
        """
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_update_step = 0
        self.initial_random_steps = initial_random_steps
        self.tau = tau
        self.gamma = gamma
        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound
        # 目标网络软更新参数

        # Actor Network (w/ Target Network)
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        # Critic Network (w/ Target Network)
        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        # Noise process
        self.noise = OUNoise(action_dim)
        self.device = device

    def sample(self, obs, add_noise=True):
        """Returns actions for given obs as per current policy."""
        if self.global_update_step < self.initial_random_steps:
            selected_action = self.env.action_space.sample()
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(obs).detach().cpu().numpy()
            self.actor.train()
            if add_noise:
                action += self.noise.sample()

            selected_action = np.clip(action, -1.0, 1.0)
            selected_action *= self.action_bound

        selected_action = selected_action.flatten()
        return selected_action

    def predict(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs).detach().cpu().numpy().flatten()
        action *= self.action_bound
        return action

    def learn(self, obs: torch.Tensor, action: torch.Tensor,
              reward: torch.Tensor, next_obs: torch.Tensor,
              terminal: torch.Tensor) -> Tuple[float, float]:
        """Update policy and value parameters using given batch of experience
        tuples.

        Q_targets = r + γ * critic_target(next_obs, actor_target(next_obs))
        where:
            actor_target(obs) -> action
            critic_target(obs, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-obs actions and Q values from target models
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            next_Q_targets = self.critic_target(next_obs, next_actions)

        # Compute Q targets for current obs (y_i)
        Q_targets = reward + self.gamma * next_Q_targets * (1 - terminal)

        # Compute critic loss
        Q_expected = self.critic(obs, action)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(obs)
        actor_loss = -torch.mean(self.critic(obs, actions_pred))
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

        self.global_update_step += 1
        return critic_loss.item(), actor_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)
