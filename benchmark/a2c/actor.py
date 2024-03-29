'''
Author: jianzhnie
Date: 2022-09-02 14:40:28
LastEditors: jianzhnie
LastEditTime: 2022-09-03 17:46:17

Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
from collections import defaultdict

import gym
import numpy as np
from atari_agent import Agent
from atari_model import ActorCritic

from rltoolkit.env.atari_wrappers import (MonitorEnv, get_wrapper_by_cls,
                                          wrap_deepmind)
from rltoolkit.env.vector_env import VectorEnv
from rltoolkit.policy.modelfree.a2c import A2C
from rltoolkit.utils.rl_utils import calc_gae


class Actor(object):

    def __init__(self, config, device):
        # the cluster may not have gpu
        self.device = device
        self.config = config
        self.envs = []
        for _ in range(config['env_num']):
            env = gym.make(config['env_name'])
            env = wrap_deepmind(env, dim=config['env_dim'], obs_format='NCHW')
            self.envs.append(env)
        self.vector_env = VectorEnv(self.envs)

        self.obs_batch = self.vector_env.reset()

        obs_shape = env.observation_space.shape
        act_dim = env.action_space.n

        self.config['obs_shape'] = obs_shape
        self.config['act_dim'] = act_dim

        model = ActorCritic(act_dim)

        model = model.to(self.device)

        algorithm = A2C(model, config)
        self.agent = Agent(algorithm, config, device=self.device)

    def sample(self):
        """Interact with the environments lambda times."""
        sample_data = defaultdict(list)

        env_sample_data = {}
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)
        for i in range(self.config['sample_batch_steps']):
            self.obs_batch = np.stack(self.obs_batch)
            action_batch, value_batch = self.agent.sample(self.obs_batch)
            next_obs_batch, reward_batch, done_batch, info_batch = self.vector_env.step(
                action_batch)

            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(
                    action_batch[env_id].item())
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])
                env_sample_data[env_id]['values'].append(
                    value_batch[env_id].item())

                if done_batch[
                        env_id] or i == self.config['sample_batch_steps'] - 1:
                    next_value = 0
                    if not done_batch[env_id]:
                        next_obs = np.expand_dims(next_obs_batch[env_id], 0)
                        next_value = self.agent.value(next_obs).item()

                    values = env_sample_data[env_id]['values']
                    rewards = env_sample_data[env_id]['rewards']
                    advantages = calc_gae(rewards, values, next_value,
                                          self.config['gamma'],
                                          self.config['lambda'])
                    target_values = advantages + values

                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['actions'].extend(
                        env_sample_data[env_id]['actions'])
                    sample_data['advantages'].extend(advantages)
                    sample_data['target_values'].extend(target_values)

                    env_sample_data[env_id] = defaultdict(list)

            self.obs_batch = next_obs_batch

        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])

        return sample_data

    def get_metrics(self):
        metrics = defaultdict(list)
        for env in self.envs:
            monitor = get_wrapper_by_cls(env, MonitorEnv)
            if monitor is not None:
                for episode_rewards, episode_steps in monitor.next_episode_results(
                ):
                    metrics['episode_rewards'].append(episode_rewards)
                    metrics['episode_steps'].append(episode_steps)
        return metrics

    def set_weights(self, params):
        self.agent.set_weights(params)
