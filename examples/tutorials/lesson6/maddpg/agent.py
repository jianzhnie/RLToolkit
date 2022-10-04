import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from rltoolkit.models.utils import soft_target_update


class PolicyNet(nn.Module):

    def __init__(self, observation_space, action_space):
        super(PolicyNet, self).__init__()
        self.num_agents = len(observation_space)
        self.action_space = action_space
        for agent_i in range(self.num_agents):
            obs_dim = observation_space[agent_i].shape[0]
            num_action = action_space[agent_i].n
            setattr(
                self, 'agent_{}'.format(agent_i),
                nn.Sequential(
                    nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 64),
                    nn.ReLU(), nn.Linear(64, num_action)))

    def forward(self, obs):
        action_logits = [
            torch.empty(1, action_space.n)
            for action_space in self.action_space
        ]
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_{}'.format(agent_i))(
                obs[:, agent_i, :]).unsqueeze(1)
            action_logits[agent_i] = x

        return torch.cat(action_logits, dim=1)


class QNet(nn.Module):

    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        for agent_i in range(self.num_agents):
            obs_dim = observation_space[agent_i].shape[0]
            setattr(
                self, 'agent_{}'.format(agent_i),
                nn.Sequential(
                    nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 64),
                    nn.ReLU(), nn.Linear(64, action_space[agent_i].n)))

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(
                obs[:, agent_i, :]).unsqueeze(1)

        # batch_size * num_agents * acion_dim
        q_value = torch.cat(q_values, dim=1)
        return q_value


class Agent(object):

    def __init__(self,
                 env: gym.Env,
                 actor_lr: float,
                 critic_lr: float,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 temperature: float = 0.005,
                 total_steps: int = int(1e4),
                 update_target_step: int = 100,
                 initial_random_steps: int = 10000,
                 device='cpu'):
        super().__init__()

        self.env = env
        self.gamma = gamma
        # 目标网络软更新参数
        self.tau = tau
        self.temperature = temperature
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.initial_random_steps = initial_random_steps

        # 策略网络
        self.actor = PolicyNet(env.observation_space,
                               env.action_space).to(device)
        # 价值网络
        self.critic = QNet(env.observation_space, env.action_space).to(device)
        # Target network
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # Create an optimizer
        # 策略网络优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device

    def sample(self, obs):
        if self.global_update_step < self.initial_random_steps:
            action = self.env.action_space.sample()
        else:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action_logits = self.actor(obs)
            action_one_hot = F.gumbel_softmax(
                logits=action_logits.squeeze(0),
                tau=self.temperature,
                hard=True)
            action = torch.argmax(action_one_hot, dim=1).data.numpy()
        return action

    def predict(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action_logits = self.actor(obs)
        action_one_hot = F.gumbel_softmax(
            logits=action_logits.squeeze(0), tau=self.temperature, hard=True)
        action = torch.argmax(action_one_hot, dim=1).data.numpy()
        return action

    def learn(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_obs: np.ndarray, terminal: np.ndarray):

        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        terminal = torch.FloatTensor(terminal).to(device)

        # pred value
        pred_q_value = self.critic(obs, action)

        with torch.no_grad():
            next_obs_action_logits = self.target_actor(next_obs)
            batch_size, n_agents, action_size = next_obs_action_logits.shape

            next_obs_action_logits = next_obs_action_logits.view(
                batch_size * n_agents, action_size)

            next_obs_action = F.gumbel_softmax(
                logits=next_obs_action_logits, tau=0.1, hard=True)

            next_obs_action = next_obs_action.view(batch_size, n_agents,
                                                   action_size)

            next_q_values = self.target_critic(next_obs, next_obs_action)

        # Target for Q regression
        # 时序差分目标
        q_target = reward + self.gamma * next_q_values * (1 - terminal)
        # 均方误差损失函数
        q_loss = F.mse_loss(pred_q_value, q_target.detach())
        # update value network
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        obs_action_logits = self.actor(obs)
        obs_action_logits = obs_action_logits.view(batch_size * n_agents,
                                                   action_size)
        obs_action = F.gumbel_softmax(
            logits=obs_action_logits, tau=0.1, hard=True)
        obs_action = obs_action.view(batch_size, n_agents, action_size)

        actor_loss = -self.critic(obs, obs_action).mean()
        # That's all for the policy loss.
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络
        soft_target_update(self.actor, self.target_actor)
        # 软更新价值网络
        soft_target_update(self.critic, self.target_critic)

        self.global_update_step += 1
        return actor_loss.item(), q_loss.item()
