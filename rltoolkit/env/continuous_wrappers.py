'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import gym
import numpy as np


class ActionMappingWrapper(gym.Wrapper):

    def __init__(self, env):
        """Map action space [-1, 1] of model output to new action space.

        [low_bound, high_bound].
        """
        gym.Wrapper.__init__(self, env)
        assert hasattr(
            self.env.action_space,
            'low'), 'action space should be instance of gym.spaces.Box'
        assert hasattr(
            self.env.action_space,
            'high'), 'action space should be instance of gym.spaces.Box'
        self.low_bound = self.env.action_space.low[0]
        self.high_bound = self.env.action_space.high[0]
        assert self.high_bound > self.low_bound
        if hasattr(env, '_max_episode_steps'):
            self._max_episode_steps = int(self.env._max_episode_steps)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, model_output_act):
        """
        Args:
            model_output_act(np.array): The values must be in in [-1, 1].
        """
        assert np.all(((model_output_act <= 1.0 + 1e-3), (model_output_act >= -1.0 - 1e-3))), \
            'the action should be in range [-1.0, 1.0]'
        mapped_action = self.low_bound + (model_output_act - (-1.0)) * (
            (self.high_bound - self.low_bound) / 2.0)
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)

        return self.env.step(mapped_action)
