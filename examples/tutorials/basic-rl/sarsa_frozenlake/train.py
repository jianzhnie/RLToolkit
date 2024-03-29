'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import time

import gym
from agent import SarsaAgent
from gridworld import FrozenLakeWapper


def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        action = agent.sample(obs)  # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
        next_action = agent.sample(next_obs)  # 根据算法选择一个动作
        # 训练 Sarsa 算法
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  # 渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    # 0 left, 1 down, 2 right, 3 up
    env = FrozenLakeWapper(env)

    agent = SarsaAgent(
        obs_dim=env.observation_space.n,
        act_dim=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1)

    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent)
        print('Episode %s: steps = %s , reward = %.1f' %
              (episode, ep_steps, ep_reward))

    # 训练结束，查看算法效果
    test_episode(env, agent)


if __name__ == '__main__':
    main()
