'''
Author: jianzhnie
Date: 2022-09-02 14:53:47
LastEditors: jianzhnie
LastEditTime: 2022-09-03 17:51:09
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

config = {

    # ==========  env config ==========
    'env_name': 'PongNoFrameskip-v4',
    'env_dim': 84,

    # ==========  actor config ==========
    'actor_num': 2,
    'env_num': 2,
    'sample_batch_steps': 20,

    # ==========  learner config ==========
    'max_sample_steps': int(1e7),
    'gamma': 0.99,
    'lambda': 1.0,

    # start learning rate
    'start_lr': 0.001,

    # coefficient of policy entropy adjustment schedule: (train_step, coefficient)
    'entropy_coeff_scheduler': [(0, -0.01)],
    'vf_loss_coeff': 0.5,
    'log_metrics_interval_s': 10,
    'learning_rate': 0.001,
}
