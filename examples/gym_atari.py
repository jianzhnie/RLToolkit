'''
Author: jianzhnie
Date: 2022-09-01 18:34:25
LastEditors: jianzhnie
LastEditTime: 2022-09-02 08:50:02
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import gym
import numpy as np


def record_episodes_video(env: gym.Env,
                          render: bool = False,
                          video_path: str = None,
                          name_prefix: str = None):

    if video_path is not None:
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env,
                                       video_folder=video_path,
                                       name_prefix=name_prefix)

    env.seed(np.random.randint(100))

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
        if done:
            env.close()
    return 0


if __name__ == '__main__':
    ENVS = [
        'PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'EnduroNoFrameskip-v4',
        'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4'
    ]
    env_id = 'PongNoFrameskip-v4'
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env,
                                   video_folder='./atari_video',
                                   name_prefix=env_id)
    env.seed(np.random.randint(100))
    obs = env.reset()
    done = False
    render = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if render:
            env.render()
        if done:
            env.close()
    for env_id in ENVS:
        env = gym.make(env_id)
        record_episodes_video(env,
                              False,
                              video_path='./atari_video',
                              name_prefix=env_id)
