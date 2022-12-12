import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

sys.path.append('../../../')
from rltoolkit.agent.base_agent import Agent
from rltoolkit.models.utils import hard_target_update
from rltoolkit.utils.utils import check_model_method


class QMixAgent(Agent):

    def __init__(self,
                 agent_model: nn.Module = None,
                 qmixer_model: nn.Module = None,
                 n_agent: int = None,
                 double_q: bool = True,
                 gamma: float = 0.99,
                 learning_rate: float = 0.0005,
                 exploration_start: float = 1.0,
                 min_exploration: float = 0.01,
                 exploration_decay: float = 0.98,
                 update_target_interval: int = 1000,
                 clip_grad_norm: bool = True,
                 device: str = 'cpu'):

        check_model_method(agent_model, 'init_hidden', self.__class__.__name__)
        check_model_method(agent_model, 'forward', self.__class__.__name__)
        check_model_method(qmixer_model, 'forward', self.__class__.__name__)
        assert hasattr(qmixer_model, 'n_agents') and not callable(
            getattr(qmixer_model, 'n_agents',
                    None)), 'qmixer_model needs to have attribute n_agents'
        assert isinstance(gamma, float)
        assert isinstance(learning_rate, float)

        self.n_agent = n_agent
        self.double_q = double_q
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.global_step = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.device = device
        self.target_update_count = 0
        self.update_target_interval = update_target_interval

        self.agent_model = agent_model
        self.qmixer_model = qmixer_model
        self.target_agent_model = deepcopy(self.agent_model)
        self.target_qmixer_model = deepcopy(self.qmixer_model)
        self.agent_model.to(device)
        self.target_agent_model.to(device)
        self.qmixer_model.to(device)
        self.target_qmixer_model.to(device)

        self.params = list(self.agent_model.parameters())
        self.params += self.qmixer_model.parameters()
        self.optimizer = torch.optim.RMSprop(
            params=self.params, lr=self.learning_rate, alpha=0.99, eps=0.00001)

    def save(self, save_dir, agent_model_name, qmixer_model_name):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        agent_model_path = os.path.join(save_dir, agent_model_name)
        qmixer_model_path = os.path.join(save_dir, qmixer_model_name)
        torch.save(self.agent_model.state_dict(), agent_model_path)
        torch.save(self.qmixer_model.state_dict(), qmixer_model_path)
        print('save model successfully!')

    def restore(self, save_dir, agent_model_name, qmixer_model_name):
        agent_model_path = os.path.join(save_dir, agent_model_name)
        qmixer_model_path = os.path.join(save_dir, qmixer_model_name)
        self.agent_model.load_state_dict(torch.load(agent_model_path))
        self.qmixer_model.load_state_dict(torch.load(qmixer_model_path))
        print('restore model successfully!')

    def reset_agent(self, batch_size=1):
        self._init_hidden_states(batch_size)

    def _init_hidden_states(self, batch_size):
        self.hidden_states = self.agent_model.init_hidden().unsqueeze(
            0).expand(batch_size, self.n_agents, -1)
        self.target_hidden_states = self.target_agent_model.init_hidden(
        ).unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def sample(self, obs, available_actions):
        ''' sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (n_agents, obs_shape)
            available_actions (np.ndarray): (n_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        '''
        epsilon = np.random.random()
        if epsilon > self.exploration:
            actions = self.predict(obs, available_actions)
        else:
            available_actions = torch.tensor(
                available_actions, dtype=torch.float32)
            actions = torch.distributions.Categorical(
                available_actions).sample().long().cpu().detach().numpy()
        self.exploration = max(self.min_exploration,
                               self.exploration - self.exploration_decay)
        return actions

    def predict(self, obs, available_actions):
        '''take greedy actions
        Args:
            obs (np.ndarray):               (n_agents, obs_shape)
            available_actions (np.ndarray): (n_agents, n_actions)
        Returns:
            actions (np.ndarray):           (n_agents, )
        '''
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        available_actions = torch.tensor(
            available_actions, dtype=torch.long, device=self.device)
        agents_q, self.hidden_states = self.agent_model(
            obs, self.hidden_states)
        # mask unavailable actions
        agents_q[available_actions == 0] = -1e10
        actions = agents_q.max(dim=1)[1].detach().cpu().numpy()
        return actions

    def update_target(self):
        hard_target_update(self.agent_model, self.target_agent_model)
        hard_target_update(self.qmixer_model, self.target_qmixer_model)

    def learn(self, state_batch, actions_batch, reward_batch, terminated_batch,
              obs_batch, available_actions_batch, filled_batch):
        '''
        Args:
            state (np.ndarray):                   (batch_size, T, state_shape)
            actions (np.ndarray):                 (batch_size, T, n_agents)
            reward (np.ndarray):                  (batch_size, T, 1)
            terminated (np.ndarray):              (batch_size, T, 1)
            obs (np.ndarray):                     (batch_size, T, n_agents, obs_shape)
            available_actions_batch (np.ndarray): (batch_size, T, n_agents, n_actions)
            filled_batch (np.ndarray):            (batch_size, T, 1)
        Returns:
            mean_loss (float): train loss
            mean_td_error (float): train TD error
        '''
        if self.global_step % self.update_target_interval == 0:
            self.update_target()
            self.target_update_count += 1

        self.global_step += 1

        state_batch = np.array(state_batch)
        actions_batch = np.array(actions_batch)
        reward_batch = np.array(reward_batch)
        terminated_batch = np.array(terminated_batch)
        obs_batch = np.array(obs_batch)
        available_actions_batch = np.array(available_actions_batch)

        state_batch = torch.tensor(
            state_batch, dtype=torch.float32, device=self.device)
        actions_batch = torch.tensor(
            actions_batch, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(
            reward_batch, dtype=torch.float32, device=self.device)
        terminated_batch = torch.tensor(
            terminated_batch, dtype=torch.float32, device=self.device)
        obs_batch = torch.tensor(
            obs_batch, dtype=torch.float32, device=self.device)
        available_actions_batch = torch.tensor(
            available_actions_batch, dtype=torch.float32, device=self.device)
        filled_batch = torch.tensor(
            filled_batch, dtype=torch.float32, device=self.device)

        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self._init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        target_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.hidden_states = self.agent_model(
                obs, self.hidden_states)
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            target_local_q, self.target_hidden_states = self.target_agent_model(
                obs, self.target_hidden_states)
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        local_qs = torch.stack(local_qs, dim=1)
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch).squeeze(3)
        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10
        if self.double_q:
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(
                dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(target_local_qs, 3,
                                               cur_max_actions).squeeze(3)
        else:
            target_local_max_qs = target_local_qs.max(
                dim=3)[0]  # idx0: value, idx1: index

        chosen_action_global_qs = self.qmixer_model(chosen_action_local_qs,
                                                    state_batch[:, :-1, :])
        target_global_max_qs = self.target_qmixer_model(
            target_local_max_qs, state_batch[:, 1:, :])

        target = reward_batch + self.gamma * (
            1 - terminated_batch) * target_global_max_qs
        td_error = target.detach() - chosen_action_global_qs
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        loss = (masked_td_error**2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()
        return loss.item(), mean_td_error.item()
