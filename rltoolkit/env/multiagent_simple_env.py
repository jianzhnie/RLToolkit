'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import multiagent.scenarios as scenarios
from gym import spaces
from multiagent.environment import MultiAgentEnv
from multiagent.multi_discrete import MultiDiscrete


class MAenv(MultiAgentEnv):
    """multiagent environment warppers for maddpg."""

    def __init__(self, scenario_name):
        env_list = [
            'simple', 'simple_adversary', 'simple_crypto', 'simple_push',
            'simple_reference', 'simple_speaker_listener', 'simple_spread',
            'simple_tag', 'simple_world_comm'
        ]
        assert scenario_name in env_list, 'Env {} not found (valid envs include {})'.format(
            scenario_name, env_list)
        # load scenario from script
        scenario = scenarios.load(scenario_name + '.py').Scenario()
        # create world
        world = scenario.make_world()
        # initial multiagent environment
        super().__init__(world, scenario.reset_world, scenario.reward,
                         scenario.observation)
        self.obs_shape_n = [
            self.get_shape(self.observation_space[i]) for i in range(self.n)
        ]
        self.act_shape_n = [
            self.get_shape(self.action_space[i]) for i in range(self.n)
        ]

    def get_shape(self, input_space):
        """
        Args:
            input_space: environment space

        Returns:
            space shape
        """
        if (isinstance(input_space, spaces.Box)):
            if (len(input_space.shape) == 1):
                return input_space.shape[0]
            else:
                return input_space.shape
        elif (isinstance(input_space, spaces.Discrete)):
            return input_space.n
        elif (isinstance(input_space, MultiDiscrete)):
            return sum(input_space.high - input_space.low + 1)
        else:
            print('[Error] shape is {}, not Box or Discrete or MultiDiscrete'.
                  format(input_space.shape))
            raise NotImplementedError
