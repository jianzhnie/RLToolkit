'''
Author: jianzhnie
Date: 2022-09-01 15:16:40
LastEditors: jianzhnie
LastEditTime: 2022-09-02 19:33:04
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.

modified from https://github.com/PaddlePaddle/rltoolkit
'''

import numpy as np
import torch

from rltoolkit.agent.base_agent import Agent
from rltoolkit.utils.scheduler import LinearDecayScheduler


class PPOAgent(Agent):

    def __init__(self, algorithm, config, device):
        super(PPOAgent, self).__init__(algorithm)

        self.config = config
        self.device = device
        if self.config['lr_decay']:
            self.lr_scheduler = LinearDecayScheduler(
                self.config['initial_lr'], self.config['num_updates'])

    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        action = self.alg.predict(obs)
        action_numpy = action.cpu().detach().numpy()
        return action_numpy

    def sample(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        value, action, action_log_probs, action_entropy = self.alg.sample(obs)

        value_numpy = value.cpu().detach().numpy()
        action_numpy = action.cpu().detach().numpy()
        action_log_probs_numpy = action_log_probs.cpu().detach().numpy()
        action_entropy_numpy = action_entropy.cpu().detach().numpy()
        return value_numpy, action_numpy, action_log_probs_numpy, action_entropy_numpy

    def value(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        value = self.alg.value(obs)
        value = value.cpu().detach().numpy()
        return value

    def learn(self, rollout):
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0
        if self.config['lr_decay']:
            lr = self.lr_scheduler.step(step_num=1)
        else:
            lr = None

        minibatch_size = int(self.config['batch_size'] //
                             self.config['num_minibatches'])

        indexes = np.arange(self.config['batch_size'])
        for epoch in range(self.config['update_epochs']):
            np.random.shuffle(indexes)
            for start in range(0, self.config['batch_size'], minibatch_size):
                end = start + minibatch_size
                sample_idx = indexes[start:end]

                batch_obs, batch_action, batch_logprob, batch_adv, batch_return, batch_value = rollout.sample_batch(
                    sample_idx)

                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_action = torch.from_numpy(batch_action).to(self.device)
                batch_logprob = torch.from_numpy(batch_logprob).to(self.device)
                batch_adv = torch.from_numpy(batch_adv).to(self.device)
                batch_return = torch.from_numpy(batch_return).to(self.device)
                batch_value = torch.from_numpy(batch_value).to(self.device)

                value_loss, action_loss, entropy_loss = self.alg.learn(
                    batch_obs, batch_action, batch_value, batch_return,
                    batch_logprob, batch_adv, lr)

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                entropy_loss_epoch += entropy_loss

        update_steps = self.config['update_epochs'] * self.config['batch_size']
        value_loss_epoch /= update_steps
        action_loss_epoch /= update_steps
        entropy_loss_epoch /= update_steps

        return value_loss_epoch, action_loss_epoch, entropy_loss_epoch, lr
