'''
Author: jianzhnie@126.com
Date: 2022-08-31 17:25:32
LastEditors: jianzhnie@126.com
LastEditTime: 2022-09-01 11:30:28
Description:

Copyright (c) 2022 by jianzhnie jianzhnie@126.com, All Rights Reserved.
'''

import glob
import os

from rltoolkit.env.gym_envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly."""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def check_model_method(model, method, algo):
    """check method existence for input model to algo.

    Args:
        model(rltoolkit.Model): model for checking
        method(str): method name
        algo(str): algorithm name

    Raises:
        AssertionError: if method is not implemented in model
    """
    if method == 'forward':
        # check if forward is overridden by the subclass
        assert callable(
            getattr(model, 'forward',
                    None)), 'forward should be a function in model class'
        assert model.forward.__func__ is not super(
            model.__class__, model
        ).forward.__func__, "{}'s model needs to implement forward method. \n".format(
            algo)
    else:
        # check if the specified method is implemented
        assert hasattr(model, method) and callable(
            getattr(
                model, method,
                None)), "{}'s model needs to implement {} method. \n".format(
                    algo, method)
