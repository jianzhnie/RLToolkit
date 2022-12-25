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

        q_vals, critic_train_stats = self.target_critic_model(
            state_batch, obs_batch, actions_batch, available_actions_batch,
            reward_batch, terminated_batch, filled_batch)

        # Calculate estimated Q-Values
        local_qs = []
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

        # Concat over time
        local_qs = torch.stack(local_qs, dim=1)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == 'pi_logits':
            local_qs_prob = torch.softmax(local_qs, dim=-1)

        # Calculated baseline
        local_qs_prob = local_qs_prob.view(-1, self.n_actions)
        q_vals = q_vals.reshape(-1, self.n_actions)
        baseline = (local_qs_prob * q_vals).sum(-1).detach()

        q_taken = torch.gather(
            q_vals, dim=1, index=actions_batch.reshape(-1, 1)).squeeze(1)
        pi_taken = torch.gather(
            local_qs_prob, dim=1, index=actions_batch.reshape(-1,
                                                              1)).squeeze(1)

        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)
        advantages = (q_taken - baseline).detach()

        entropy = -torch.sum(
            local_qs_prob * torch.log(local_qs_prob + 1e-10), dim=-1)
        coma_loss = -(
            (advantages * log_pi_taken + self.entropy_coef * entropy) *
            mask).sum() / mask.sum()

        # Optimise
        self.actor_optimizer.zero_grad()
        coma_loss.backward()
        if self.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(self.actor_parameters,
                                           self.grad_norm_clip)
        self.actor_optimizer.step()

        return coma_loss.item()

    def _train_critic(self, state_batch, actions_batch, reward_batch,
                      terminated_batch, obs_batch, available_actions_batch,
                      filled_batch):

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        # Optimise critic
        with torch.no_grad():
            target_q_vals = self.critic_model(state_batch)

        targets_taken = torch.gather(
            target_q_vals, dim=3, index=actions_batch).squeeze(3)

        targets = self.nstep_returns(reward_batch, terminated_batch,
                                     targets_taken, self.q_nstep)

        actions = actions[:, :-1]
        q_vals = self.critic_model(state_batch)[:, :-1]
        q_taken = torch.gather(q_vals, dim=3, index=actions).squeeze(3)

        td_error = (q_taken - targets.detach())
        masked_td_error = td_error * mask

        loss = (masked_td_error**2).sum() / mask.sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters,
                                       self.grad_norm_clip)
        self.critic_optimizer.step()

        return q_vals

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = torch.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = torch.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.gamma**step * values[:, t] * mask[:,
                                                                             t]
                elif t == rewards.size(1) - 1 and self.add_value_last_step:
                    nstep_return_t += self.gamma**step * rewards[:,
                                                                 t] * mask[:,
                                                                           t]
                    nstep_return_t += self.gamma**(step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.gamma**(
                        step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values
