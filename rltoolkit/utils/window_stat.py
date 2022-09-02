'''
Author: jianzhnie
Date: 2022-09-02 14:49:20
LastEditors: jianzhnie
LastEditTime: 2022-09-02 14:49:45
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
__all__ = ['WindowStat']

import numpy as np


class WindowStat(object):
    """Tool to maintain statistical data in a window."""

    def __init__(self, window_size):
        self.items = [None] * window_size
        self.idx = 0
        self.count = 0

    def add(self, obj):
        self.items[self.idx] = obj
        self.idx += 1
        self.count += 1
        self.idx %= len(self.items)

    @property
    def mean(self):
        if self.count > 0:
            return np.mean(self.items[:self.count])
        else:
            return None

    @property
    def min(self):
        if self.count > 0:
            return np.min(self.items[:self.count])
        else:
            return None

    @property
    def max(self):
        if self.count > 0:
            return np.max(self.items[:self.count])
        else:
            return None
