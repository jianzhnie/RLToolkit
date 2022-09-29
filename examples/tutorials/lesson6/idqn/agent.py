import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class QNet(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 hiddend_dim: int = 128):
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

        return torch.cat(q_values, dim=1)


class Agent(object):

    def __init__(self,
                 env: gym.Env,
                 algo: str = 'dqn',
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.1,
                 learning_rate: float = 0.001,
                 update_target_step: int = 100,
                 device='cpu'):
        super().__init__()

        self.algo = algo
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.global_update_step = 0
        self.update_target_step = update_target_step

        # Main network
        self.qnet = QNet(env.observation_space, env.action_space).to(device)
        # Target network
        self.target_qnet = copy.deepcopy(self.qnet)
        # Create an optimizer
        self.optimizer = Adam(self.qnet.parameters(), lr=learning_rate)

        self.device = device

    def sample(self, obs):
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        out = self.qnet(obs)
        mask = (torch.rand((out.shape[0], )) <= self.epsilon)
        action = torch.empty((
            out.shape[0],
            out.shape[1],
        ))
        action[mask] = torch.randint(0, out.shape[2],
                                     action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        action = action[0].cpu().numpy().tolist()
        return action

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        out = self.qnet(obs)
        action = torch.empty((
            out.shape[0],
            out.shape[1],
        ))
        action = out.argmax(dim=2).float()
        action = action[0].cpu().numpy().tolist()
        return action

    def learn(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_obs: np.ndarray, terminal: np.ndarray):

        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        terminal = torch.FloatTensor(terminal).to(device)

        pred_value = self.qnet(obs).gather(
            2,
            action.unsqueeze(-1).long()).squeeze(-1)
        next_q_value = self.target_qnet(next_obs).max(dim=2)[0]

        target = reward + self.gamma * next_q_value * terminal
        loss = F.smooth_l1_loss(pred_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
