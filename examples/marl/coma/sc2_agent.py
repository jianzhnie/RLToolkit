'''
Author: jianzhnie
Date: 2022-09-03 11:00:14
LastEditors: jianzhnie
LastEditTime: 2022-09-03 11:05:51
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import RMSprop

from rltoolkit.models.utils import check_model_method


class Agents(object):

    def __init__(self,
                 agent_model: nn.Module = None,
                 critic_model: nn.Module = None,
                 n_agents: int = None,
                 n_actions: int = None,
                 actor_lr: float = None,
                 critic_lr: float = None,
                 gamma: float = None,
                 td_lambda: float = None,
                 agent_output_type: str = 'pi_logits',
                 grad_norm_clip: float = None,
                 optim_alpha: float = 0.99,
                 optim_eps: float = 0.00001,
                 device: str = 'cpu'):
        """COMA Agent.

        Args:
            model: forward network of actor and critic.
            n_actions (int): action dim for each agent
            n_agents (int): agents number
            grad_norm_clip (int or float): gradient clip, prevent gradient explosion
            actor_lr (float): actor network learning rate
            critic_lr (float): critic network learning rate
            gamma (float):  discounted factor for reward computation
            td_lambda (float): lambda of td-lambda return
        """
        # checks
        check_model_method(agent_model, 'forward', self.__class__.__name__)
        check_model_method(critic_model, 'forward', self.__class__.__name__)
        assert isinstance(n_agents, int)
        assert isinstance(n_actions, int)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        assert isinstance(gamma, float)
        assert isinstance(td_lambda, float)
        assert isinstance(grad_norm_clip, int) or isinstance(
            grad_norm_clip, float)

        self.n_actions = n_actions
        self.n_agents = n_agents
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.agent_output_type = agent_output_type
        self.grad_norm_clip = grad_norm_clip

        self.agent_model = agent_model.to(device)
        self.target_agent_model = deepcopy(agent_model).to(device)
        self.critic_model = critic_model.to(device)
        self.target_critic_model = deepcopy(critic_model).to(device)

        self.actor_parameters = list(self.agent_model.parameters())
        self.critic_parameters = list(self.critic_model.parameters())

        self.actor_optimizer = RMSprop(
            self.actor_parameters,
            lr=self.actor_lr,
            alpha=optim_alpha,
            eps=optim_eps)

        self.critic_optimizer = RMSprop(
            self.critic_parameters,
            lr=self.critic_lr,
            alpha=optim_alpha,
            eps=optim_eps)

    def init_hidden(self, ep_num=1):
        """ function: init a hidden tensor for every agent
            input:
                ep_num: How many episodes are included in a batch of data
            output:
                rnn_h: rnn hidden state, shape (ep_num, n_agents, hidden_size)
        """
        assert hasattr(self.model.actor_model, 'init_hidden'), \
            "actor must have rnn structure and has method 'init_hidden' to make hidden states"
        rnn_h = self.model.actor_model.init_hidden().unsqueeze(0).expand(
            ep_num, self.n_agents, -1)
        return rnn_h

    def predict(self, obs, rnn_h_in):
        """input:
                obs: obs + last_action + agent_id, shape: (1, obs_shape + n_actions + n_agents)
                rnn_h_in: rnn's hidden input
            output:
                prob: output of actor, shape: (1, n_actions)
                rnn_h_out: rnn's hidden output
        """
        obs = np.expand_dims(obs, 0)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            policy_logits, rnn_h_out = self.model.policy(obs, rnn_h_in)
            # input obs shape [1, 42]
            prob = torch.nn.functional.softmax(policy_logits, dim=-1)
            # shape [1, 9]
        return prob, rnn_h_out

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

        # set the actions to torch.Long
        actions_batch = actions_batch.to(self.device, dtype=torch.long)
        # get the batch_size and episode_length
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        # Calculate estimated Q-Values
        local_qs = []
        target_local_qs = []
        self._init_hidden_states(batch_size)
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs: (batch_size * n_agents, obs_shape)
            obs = obs.reshape(-1, obs_batch.shape[-1])
            # Calculate estimated Q-Values
            local_q, self.hidden_states = self.agent_model(
                obs, self.hidden_states)
            #  local_q: (batch_size * n_agents, n_actions) -->  (batch_size, n_agents, n_actions)
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            # Calculate the Q-Values necessary for the target
            target_local_q, self.target_hidden_states = self.target_agent_model(
                obs, self.target_hidden_states)
            # target_local_q: (batch_size * n_agents, n_actions) -->  (batch_size, n_agents, n_actions)
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        # Concat over time
        local_qs = torch.stack(local_qs, dim=1)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == 'pi_logits':
            local_qs_prob = torch.softmax(local_qs, dim=-1)

        local_qs_prob[available_actions_batch == 0] = 0.0

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch).squeeze(3)

        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10

        # Max over target Q-Values
        if self.double_q:
            # Get actions that maximise live Q (for double q-learning)
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(
                dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, dim=3, index=cur_max_actions).squeeze(3)
        else:
            # idx0: value, idx1: index
            target_local_max_qs = target_local_qs.max(dim=3)[0]

        # Mixing network
        # mix_net, input: ([Q1, Q2, ...], state), output: Q_total
        if self.mixer_model is not None:
            chosen_action_global_qs = self.mixer_model(chosen_action_local_qs,
                                                       state_batch[:, :-1, :])
            target_global_max_qs = self.target_mixer_model(
                target_local_max_qs, state_batch[:, 1:, :])

        # Calculate 1-step Q-Learning targets
        target = reward_batch + self.gamma * (
            1 - terminated_batch) * target_global_max_qs
        #  Td-error
        td_error = target.detach() - chosen_action_global_qs
        #  0-out the targets that came from padded data
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()

        return loss.item(), mean_td_error.item()
