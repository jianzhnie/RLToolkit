'''
Author: jianzhnie
Date: 2022-09-01 18:34:25
LastEditors: jianzhnie
LastEditTime: 2022-09-02 08:50:02
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import gym

# environment_name = 'ALE/PongNoFrameskip-v4 '
# env = gym.make(environment_name)
env = gym.make('PongNoFrameskip-v4', render_mode='human')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
env.close()  # https://github.com/openai/gym/issues/893
