import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from rltoolkit.models.utils import hard_target_update
from rltoolkit.utils import LinearDecayScheduler


class QNet(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_size=32,
                 recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.hidden_size = hidden_size
        self.recurrent = recurrent
        for agent_i in range(self.num_agents):
            obs_dim = observation_space[agent_i].shape[0]
            setattr(
                self, 'agent_feature_{}'.format(agent_i),
                nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(),
                              nn.Linear(64, hidden_size)))

            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i),
                        nn.GRUCell(hidden_size, hidden_size))
            setattr(self, 'agent_q_{}'.format(agent_i),
                    nn.Linear(hidden_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hidden_size)
                       ] * self.num_agents
        for agent_i in range(self.num_agents):
            x = getattr(self,
                        'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x = getattr(self,
                            'agent_gru_{}'.format(agent_i))(x,
                                                            hidden[:,
                                                                   agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = getattr(
                self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)

        # batch_size * num_agents * acion_dim
        q_value = torch.cat(q_values, dim=1)
        hidden = torch.cat(next_hidden, dim=1)

        return q_value, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hidden_size))


class Agent(object):

    def __init__(self,
                 env: gym.Env,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 chunk_size: int = 10,
                 recurrent: bool = False,
                 grad_clip_norm: float = 5.0,
                 total_steps: int = int(1e4),
                 update_target_step: int = 100,
                 device='cpu'):
        super().__init__()

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.curr_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self._chunk_size = chunk_size if recurrent else 1

        # Main network
        self.qnet = QNet(env.observation_space,
                         env.action_space,
                         recurrent=recurrent).to(device)
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
        hidden = self.qnet.init_hidden()
        q_value, hidden = self.qnet(obs, hidden)
        action = q_value.argmax(dim=2)
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
        terminal = terminal.bool().all(dim=2, keepdims=True).float()

        hidden = self.qnet.init_hidden(self.batch_size)
        target_hidden = self.target_qnet.init_hidden(self.batch_size)

        loss = 0
        for step_i in range(self._chunk_size):
            q_out, hidden = self.qnet(obs[:, step_i, :, :], hidden)

            q_action = q_out.gather(
                2, action[:, step_i, :].unsqueeze(-1)).squeeze(-1)

            sum_q = q_action.sum(dim=1, keepdims=True)

            next_q_value, target_hidden = self.target_qnet(
                next_obs[:, step_i, :, :], target_hidden.detach())

            next_q_value = next_q_value.max(dim=2)[0].squeeze(-1)

            target_q = reward[:, step_i, :].sum(dim=1, keepdims=True)

            target_q += self.gamma * next_q_value.sum(
                dim=1, keepdims=True) * (1 - terminal[:, step_i])

            loss += F.smooth_l1_loss(sum_q, target_q.detach())

            done_mask = terminal[:, step_i].squeeze(-1).bool()
            hidden[done_mask] = self.qnet.init_hidden(len(hidden[done_mask]))
            target_hidden[done_mask] = self.target_qnet.init_hidden(
                len(target_hidden[done_mask]))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(),
                                 self.grad_clip_norm,
                                 norm_type=2)
        self.optimizer.step()

        self.global_update_step += 1
