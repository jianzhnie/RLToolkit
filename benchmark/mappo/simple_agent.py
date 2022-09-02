'''
Author: jianzhnie
Date: 2022-09-02 15:02:03
LastEditors: jianzhnie
LastEditTime: 2022-09-02 15:20:37
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import numpy as np
import torch

from rltoolkit.core.agent import Agent


class SimpleAgent(Agent):

    def __init__(self, algorithm):
        self.value_normalizer = algorithm.value_normalizer
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        super(SimpleAgent, self).__init__(algorithm)

    def sample(self, share_obs, obs):
        obs = torch.from_numpy(obs).to(self.device)
        share_obs = torch.from_numpy(share_obs).to(self.device)
        value, action, action_log_prob = self.alg.sample(share_obs, obs)
        return value.detach().cpu().numpy(), action.detach().cpu().numpy(
        ), action_log_prob.detach().cpu().numpy()

    def learn(self, buffer, ppo_epoch, num_mini_batch, use_popart):
        if use_popart:
            advantages = buffer.returns[:
                                        -1] - self.value_normalizer.denormalize(
                                            buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0

        for _ in range(ppo_epoch):
            data_generator = buffer.sample_batch(advantages, num_mini_batch)
            for sample in data_generator:
                share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch,\
                     masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ = sample

                share_obs_batch = torch.from_numpy(share_obs_batch).to(
                    self.device)
                obs_batch = torch.from_numpy(obs_batch).to(self.device)
                actions_batch = torch.from_numpy(actions_batch).to(self.device)
                value_preds_batch = torch.from_numpy(value_preds_batch).to(
                    self.device)
                return_batch = torch.from_numpy(return_batch).to(self.device)
                active_masks_batch = torch.from_numpy(active_masks_batch).to(
                    self.device)
                old_action_log_probs_batch = torch.from_numpy(
                    old_action_log_probs_batch).to(self.device)
                adv_targ = torch.from_numpy(adv_targ).to(self.device)

                value_loss, policy_loss, dist_entropy = self.alg.learn(
                    share_obs_batch, obs_batch, actions_batch,
                    value_preds_batch, return_batch,
                    old_action_log_probs_batch, adv_targ, active_masks_batch)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()

        num_updates = ppo_epoch * num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def value(self, share_obs):
        share_obs = torch.from_numpy(share_obs).to(self.device)
        next_values = self.alg.value(share_obs).detach().cpu().numpy()
        return next_values
