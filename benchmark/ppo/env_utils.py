'''
Author: jianzhnie
Date: 2022-09-01 15:27:01
LastEditors: jianzhnie
LastEditTime: 2022-09-01 15:53:58
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import gym
import numpy as np

from rltoolkit.env.atari_wrappers import wrap_deepmind
from rltoolkit.env.mujoco_wrappers import wrap_rms
from rltoolkit.utils import logger

TEST_EPISODE = 3
# wrapper parameters for atari env
ENV_DIM = 84
OBS_FORMAT = 'NCHW'
# wrapper parameters for mujoco env
GAMMA = 0.99


class ParallelEnv(object):

    def __init__(self, config=None):
        self.config = config
        self.env_num = config['env_num']
        base_env = LocalEnv
        if config['seed']:
            self.env_list = [
                base_env(config['env'], config['seed'] + i)
                for i in range(self.env_num)
            ]
        else:
            self.env_list = [
                base_env(config['env']) for _ in range(self.env_num)
            ]

        self._max_episode_steps = self.env_list[0]._max_episode_steps

        self.total_steps = 0
        self.episode_steps_list = [0] * self.env_num
        self.episode_reward_list = [0] * self.env_num
        # used for env initialization for evaluating in mujoco environment
        self.eval_ob_rms = None

    def reset(self):
        obs_list = [env.reset() for env in self.env_list]
        self.obs_list = np.array(obs_list)
        return self.obs_list

    def step(self, action_list):
        next_obs_list, reward_list, done_list, info_list = [], [], [], []

        for i in range(self.env_num):
            self.total_steps += 1

            next_obs, reward, done, info = self.env_list[i].step(
                action_list[i])

            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += reward

            if done or self.episode_steps_list[i] >= self._max_episode_steps:

                next_obs = self.env_list[i].reset()
                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0
                if self.env_list[i].continuous_action:
                    # get running mean and variance of obs
                    self.eval_ob_rms = self.env_list[i].env.get_ob_rms()

            next_obs_list.append(next_obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return np.array(next_obs_list), np.array(reward_list), np.array(
            done_list), np.array(info_list)


class LocalEnv(object):

    def __init__(self, env_name, env_seed=None, test=False, ob_rms=None):
        env = gym.make(env_name)
        self._max_episode_steps = env._max_episode_steps

        # is instance of gym.spaces.Box
        if hasattr(env.action_space, 'high'):
            self.continuous_action = True
            if test:
                self.env = wrap_rms(env, GAMMA, test=True, ob_rms=ob_rms)
            else:
                self.env = wrap_rms(env, gamma=GAMMA)
        # is instance of gym.spaces.Discrete
        elif hasattr(env.action_space, 'n'):
            self.continuous_action = False
            if test:
                self.env = wrap_deepmind(
                    env,
                    dim=ENV_DIM,
                    obs_format=OBS_FORMAT,
                    test=True,
                    test_episodes=1)
            else:
                self.env = wrap_deepmind(
                    env, dim=ENV_DIM, obs_format=OBS_FORMAT)
        else:
            raise AssertionError(
                'act_space must be instance of gym.spaces.Box or gym.spaces.Discrete'
            )

        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

        if env_seed:
            self.env.seed(env_seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class RemoteEnv(object):

    def __init__(self, env_name, env_seed=None, test=False, ob_rms=None):
        env = gym.make(env_name)
        self._max_episode_steps = env._max_episode_steps

        if hasattr(env.action_space, 'high'):
            self.continuous_action = True
            if test:
                self.env = wrap_rms(env, GAMMA, test=True, ob_rms=ob_rms)
            else:
                self.env = wrap_rms(env, gamma=GAMMA)
        elif hasattr(env.action_space, 'n'):
            self.continuous_action = False
            if test:
                self.env = wrap_deepmind(
                    env,
                    dim=ENV_DIM,
                    obs_format=OBS_FORMAT,
                    test=True,
                    test_episodes=1)
            else:
                self.env = wrap_deepmind(
                    env, dim=ENV_DIM, obs_format=OBS_FORMAT)
        else:
            raise AssertionError(
                'act_space must be instance of gym.spaces.Box or gym.spaces.Discrete'
            )
        if env_seed:
            self.env.seed(env_seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return logger.warning(
            'Can not render in remote environment, render() have been skipped.'
        )
