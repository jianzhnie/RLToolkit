'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
# -*- coding: utf-8 -*-

import numpy as np


class SarsaAgent(object):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 learning_rate=0.01,
                 gamma=0.9,
                 epsilon=0.1):
        self.act_dim = act_dim  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = epsilon  # 按一定概率随机选动作
        self.Q = np.zeros((obs_dim, act_dim))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.act_dim)  # 有一定概率随机探索选取一个动作
        else:  # 根据table的Q值选动作
            action = self.predict(obs)
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]
        # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q

    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
