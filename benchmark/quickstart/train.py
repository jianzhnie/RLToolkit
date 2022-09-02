'''
Author: jianzhnie
Date: 2022-09-02 12:23:30
LastEditors: jianzhnie
LastEditTime: 2022-09-02 12:29:55
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import sys

import gym
import numpy as np
import torch

sys.path.append('../../')
from cartpole_agent import CartpoleAgent
from cartpole_model import CartpoleModel

from rltoolkit.policy.modelfree.policy_gradient import PolicyGradient
from rltoolkit.utils import logger

OBS_DIM = 4
ACT_DIM = 2
LEARNING_RATE = 1e-3


def run_episode(env, agent, train_or_test='train'):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        if train_or_test == 'train':
            action = agent.sample(obs)
        else:
            action = agent.predict(obs)
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


def calc_reward_to_go(reward_list):
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += reward_list[i + 1]
    return np.array(reward_list)


def main():
    env = gym.make('CartPole-v0')
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CartpoleModel(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    alg = PolicyGradient(model, LEARNING_RATE, device=device)
    agent = CartpoleAgent(alg, device=device)

    for i in range(1000):  # 1000 episodes
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            logger.info('Episode {}, Reward Sum {}.'.format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            _, _, reward_list = run_episode(env, agent, train_or_test='test')
            total_reward = np.sum(reward_list)
            logger.info('Test reward: {}'.format(total_reward))


if __name__ == '__main__':
    main()
