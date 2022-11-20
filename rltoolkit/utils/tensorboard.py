'''
Author: jianzhnie
Date: 2022-09-01 17:33:14
LastEditors: jianzhnie
LastEditTime: 2022-09-01 17:34:29
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

from tensorboardX import SummaryWriter

from rltoolkit.utils import logger

__all__ = []

_writer = None
_WRITTER_METHOD = ['add_scalar', 'add_histogram', 'close', 'flush']


def create_file_after_first_call(func_name):

    def call(*args, **kwargs):
        global _writer
        if _writer is None:
            logdir = logger.get_dir()
            if logdir is None:
                logdir = logger.auto_set_dir(action='d')
            _writer = SummaryWriter(logdir=logger.get_dir())
        func = getattr(_writer, func_name)
        func(*args, **kwargs)
        _writer.flush()

    return call


# export writer functions
for func_name in _WRITTER_METHOD:
    locals()[func_name] = create_file_after_first_call(func_name)
    __all__.append(func_name)
