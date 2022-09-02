'''
Author: jianzhnie
Date: 2022-09-01 15:34:45
LastEditors: jianzhnie
LastEditTime: 2022-09-02 10:14:28
Description:
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

atari_config = {
    # Commented parameters are set to default values in ppo

    # ==========  env config ==========
    'env': 'PongNoFrameskip-v4',  # environment name
    'continuous_action': False,  # action type of the environment
    'env_num': 4,  # number of the environment
    'seed': None,  # seed of the experiment

    # ==========  training config ==========
    'train_total_steps': int(1e7),  # max training steps
    'step_nums': 128,  # data collecting time steps (ie. T in the paper)
    'num_minibatches': 4,  # number of training minibatches per update.
    'update_epochs': 4,  # number of epochs for updating (ie K in the paper)
    'eval_episode': 3,
    'test_every_steps': int(5e3),  # interval between evaluations

    # ========== coefficient of ppo ==========
    'initial_lr': 2.5e-4,  # start learning rate
    'lr_decay': True,  # whether or not to use linear decay rl
    'eps': 1e-5,  # Adam optimizer epsilon (default: 1e-5)
    'clip_param': 0.1,  # epsilon in clipping loss
    'entropy_coef': 0.01,  # Entropy coefficient (ie. c_2 in the paper)
    'value_loss_coef': 0.5,  # Value loss coefficient (ie. c_1 in the paper)
    'max_grad_norm': 0.5,  # Max gradient norm for gradient clipping
    'use_clipped_value_loss': True,  # advantages normalization
    'clip_vloss': True,
    # whether or not to use a clipped loss for the value function
    'gamma': 0.99,  # discounting factor
    'gae': True,  # whether or not to use GAE
    'gae_lambda': 0.95,  # Lambda parameter for calculating N-step advantage
}
