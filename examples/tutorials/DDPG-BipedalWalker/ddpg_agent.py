import copy
from typing import Any, Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
from model import Actor, Critic
from torch.optim import Adam

from rltoolkit.models.ounoise import OUNoise
from rltoolkit.models.utils import soft_target_update


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
                 ou_noise_theta: float,
                 ou_noise_sigma: float,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 update_target_step: int = 1,
                 device: Any = None):
        """Initialize an Agent object.

        Params
        ======
            obs_dim (int): dimension of each obs
            action_dim (int): dimension of each action
        """
        self.env = env
        self.action_dim = action_dim
        # 目标网络软更新参数
        self.tau = tau
        self.gamma = gamma
        # action_bound是环境可以接受的动作最大值
        self.action_bound = action_bound
        self.global_update_step = 0
        self.update_target_step = update_target_step

        # Actor Network (w/ Target Network)
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        # Critic Network (w/ Target Network)
        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        # Noise process
        self.noise = OUNoise(
            mu=0.0, sigma=ou_noise_sigma, theta=ou_noise_theta)
        self.device = device

    def sample(self, obs, add_noise=True):
        """Returns actions for given obs as per current policy."""
        obs = torch.from_numpy(obs).float().to(self.device)
        action = self.actor(obs).detach().cpu().numpy()
        if add_noise:
            action += self.noise(action.shape)
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
        critic_loss = F.mse_loss(Q_expected, Q_targets)

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
        soft_target_update(self.critic, self.critic_target, self.tau)
        soft_target_update(self.actor, self.actor_target, self.tau)

        self.global_update_step += 1
        return critic_loss.item(), actor_loss.item()
