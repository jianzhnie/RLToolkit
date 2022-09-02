'''
Author: jianzhnie
Date: 2022-09-02 14:49:20
LastEditors: jianzhnie
LastEditTime: 2022-09-02 14:49:35
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import time

from .window_stat import WindowStat

__all_ = ['TimeStat']


class TimeStat(object):
    """A time stat for logging the elapsed time of code running.

    Example:
        time_stat = TimeStat()
        with time_stat:
            // some code
        print(time_stat.mean)
    """

    def __init__(self, window_size=1):
        self.time_samples = WindowStat(window_size)
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        time_delta = time.time() - self._start_time
        self.time_samples.add(time_delta)

    @property
    def mean(self):
        return self.time_samples.mean

    @property
    def min(self):
        return self.time_samples.min

    @property
    def max(self):
        return self.time_samples.max
