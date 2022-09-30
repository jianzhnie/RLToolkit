import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from rltoolkit.models.utils import hard_target_update
from rltoolkit.utils.scheduler import LinearDecayScheduler


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
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 learning_rate: float = 0.001,
                 total_steps: int = int(1e4),
                 update_target_step: int = 100,
                 device='cpu'):
        super().__init__()

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.curr_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.global_update_step = 0
        self.update_target_step = update_target_step

        # Main network
        self.qnet = QNet(env.observation_space, env.action_space).to(device)
        # Target network
        self.target_qnet = copy.deepcopy(self.qnet)
        # Create an optimizer
        self.optimizer = Adam(self.qnet.parameters(), lr=learning_rate)
        self.ep_scheduler = LinearDecayScheduler(epsilon, total_steps)

        self.device = device

    def sample(self, obs):

        if np.random.rand() <= self.curr_epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.predict(obs)

        # Decaying epsilon
        self.curr_epsilon = max(self.ep_scheduler.step(1), self.min_epsilon)
        return action

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        out = self.qnet(obs)
        action = out.argmax(dim=2)
        action = action[0].cpu().numpy().tolist()
        return action

    def learn(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_obs: np.ndarray, terminal: np.ndarray):

        if self.global_update_step % self.update_target_step == 0:
            hard_target_update(self.qnet, self.target_qnet)

        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        terminal = torch.FloatTensor(terminal).to(device)

        # Prediction Q(s)
        pred_value = self.qnet(obs).gather(2, action.unsqueeze(-1)).squeeze(-1)

        # Target for Q regression
        next_q_value = self.target_qnet(next_obs).max(dim=2)[0]

        target = reward + self.gamma * next_q_value * (1 - terminal)
        loss = F.mse_loss(pred_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_update_step += 1
