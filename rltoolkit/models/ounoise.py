import copy
import random
from typing import Optional, Sequence, Union

import numpy as np


class OUNoise_old(object):
    """Ornstein-Uhlenbeck process.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self,
                 size: int,
                 mu: float = 0.0,
                 theta: float = 0.15,
                 sigma: float = 0.2):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class OUNoise(object):

    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 0.3,
                 theta: float = 0.15,
                 dt: float = 1e-2,
                 x0: Optional[Union[float, np.ndarray]] = None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __call__(self, size: Sequence[int]):
        x = self.x_prev + self.theta * (
            self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
                self.dt) * np.random.normal(size=size)
        self.x_prev = x
        return x
