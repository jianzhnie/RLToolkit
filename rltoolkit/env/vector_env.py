'''
Author: jianzhnie
Date: 2022-09-01 15:29:39
LastEditors: jianzhnie
LastEditTime: 2022-09-01 15:29:48
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import six

__all__ = ['VectorEnv']


class VectorEnv(object):
    """vector of envs to support vector reset and vector step.

    `vector_step` api will automatically reset envs which are done.
    """

    def __init__(self, envs):
        """
        Args:
            envs: List of env
        """
        self.envs = envs
        self.envs_num = len(envs)

    def reset(self):
        """
        Returns:
            List of obs
        """
        return [env.reset() for env in self.envs]

    def step(self, actions):
        """
        Args:
            actions: List or array of action

        Returns:
            obs_batch: List of next obs of envs
            reward_batch: List of return reward of envs
            done_batch: List of done of envs
            info_batch: List of info of envs
        """
        obs_batch, reward_batch, done_batch, info_batch = [], [], [], []
        for env_id in six.moves.range(self.envs_num):
            obs, reward, done, info = self.envs[env_id].step(actions[env_id])

            if done:
                obs = self.envs[env_id].reset()

            obs_batch.append(obs)
            reward_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)
        return obs_batch, reward_batch, done_batch, info_batch
