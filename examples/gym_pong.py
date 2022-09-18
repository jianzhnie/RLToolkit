'''
Author: jianzhnie
Date: 2022-09-01 18:34:25
LastEditors: jianzhnie
LastEditTime: 2022-09-02 08:50:02
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import gym

if __name__ == '__main__':
    env = gym.make('PongNoFrameskip-v4', render_mode='human')
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()  # https://github.com/openai/gym/issues/893
