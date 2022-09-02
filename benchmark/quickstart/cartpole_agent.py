'''
Author: jianzhnie
Date: 2022-09-02 12:21:35
LastEditors: jianzhnie
LastEditTime: 2022-09-02 19:32:55
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import numpy as np
import torch

from rltoolkit.agent.base_agent import Agent


class CartpoleAgent(Agent):
    """Agent of Cartpole env.

    Args:
        algorithm(parl.Algorithm): algorithm used to solve the problem.
    """

    def __init__(self, algorithm, device):
        super(CartpoleAgent, self).__init__(algorithm)
        self.device = device

    def sample(self, obs):
        """Sample an action when given an observation.

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            action(int)
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        prob = self.alg.predict(obs).cpu()
        prob = prob.data.numpy()
        action = np.random.choice(len(prob), 1, p=prob)[0]
        return action

    def predict(self, obs):
        """Predict an action when given an observation.

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            action(int)
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        prob = self.alg.predict(obs)
        _, action = prob.max(-1)
        return action.item()

    def learn(self, obs, action, reward):
        """Update model with an episode data.

        Args:
            obs(np.float32): shape of (batch_size, obs_dim)
            action(np.int64): shape of (batch_size)
            reward(np.float32): shape of (batch_size)

        Returns:
            loss(float)
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)

        loss = self.alg.learn(obs, action, reward)
        return loss.item()
