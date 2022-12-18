'''
Author: jianzhnie@126.com
Date: 2022-08-31 17:25:32
LastEditors: jianzhnie@126.com
LastEditTime: 2022-09-01 11:30:28
Description:

Copyright (c) 2022 by jianzhnie jianzhnie@126.com, All Rights Reserved.
'''

import gym

from rltoolkit.env.gym_envs import VecNormalize


def make_env(env_id, seed, idx, capture_video, run_name):

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None
