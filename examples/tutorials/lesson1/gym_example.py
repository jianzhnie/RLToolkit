'''
Author: jianzhnie
Date: 2022-08-14 08:35:09
LastEditors: jianzhnie
LastEditTime: 2022-09-01 18:19:23
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import gym  # 导入 Gym 的 Python 接口环境包

if __name__ == '__main__':
    env = gym.make('CartPole-v1')  # 构建实验环境
    # 重置一个回合
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 从动作空间中随机选取一个动作
        # print(action)
        observation, reward, done, info = env.step(action)  # 用于提交动作，括号内是具体的动作
        # print(observation, reward, done, info)
        env.render()  # 显示图形界面
        if done:
            observation = env.reset()
    env.close()  # 关闭环境
