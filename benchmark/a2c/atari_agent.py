'''
Author: jianzhnie
Date: 2022-09-02 14:38:36
LastEditors: jianzhnie
LastEditTime: 2022-09-02 14:38:39
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import numpy as np
import torch

from rltoolkit.core.agent import Agent
from rltoolkit.utils.scheduler import LinearDecayScheduler, PiecewiseScheduler

torch.set_num_threads(1)


class Agent(Agent):

    def __init__(self, algorithm, config):
        super(Agent, self).__init__(algorithm)
        self.obs_shape = config['obs_shape']
        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_sample_steps'])

        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def sample(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        probs, values = self.alg.prob_and_value(obs)
        probs = probs.cpu().detach().numpy()
        values = values.cpu().detach().numpy()
        sample_actions = np.array(
            [np.random.choice(len(prob), 1, p=prob)[0] for prob in probs])
        return sample_actions, values

    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        predict_actions = self.alg.predict(obs)
        return predict_actions.cpu().detach().numpy()

    def value(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        values = self.alg.value(obs)
        return values

    def learn(self, obs, actions, advantages, target_values):
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        target_values = torch.FloatTensor(target_values).to(self.device)

        lr = self.lr_scheduler.step(step_num=obs.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()
        total_loss, pi_loss, vf_losss, entropy = self.alg.learn(
            obs, actions, advantages, target_values, lr, entropy_coeff)

        return total_loss.cpu().detach().numpy(), pi_loss.cpu().detach().numpy(), \
            vf_losss.cpu().detach().numpy(), entropy.cpu().detach().numpy(), lr, entropy_coeff
