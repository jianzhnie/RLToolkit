# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import sys
import time

import gym
import torch
from agent import Agent

sys.path.append('../../../../')
from rltoolkit.utils.utils import make_env

config = {
    'seed': 42,
    'env': 'CartPole-v0',
    'num_processes': 50,
    'algo': 'dqn',
    'use_wandb': False,
    'capture_video': False,
    'hidden_dim': 128,
    'total_steps': 5000,  # max training steps
    'memory_size': 10000,  # Replay buffer size
    'memory_warmup_size': 1000,  # Replay buffer memory_warmup_size
    'batch_size': 32,  # repaly sample batch size
    'update_target_step': 100,  # target model update freq
    'learning_rate': 0.001,  # start learning rate
    'epsilon': 1,  # start greedy epsilon
    'epsilon_decay': 0.97,  # epsilon decay rate
    'min_epsilon': 0.1,
    'gamma': 0.99,  # discounting factor
    'eval_render': True,  # do eval render
    'train_log_interval': 1,
    'test_log_interval': 5,  # evaluation freq
    'log_dir': 'work_dir',
    'video_folder': 'results'
}


def main():
    args = argparse.Namespace(**config)
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    run_name = f'{args.env}_{args.algo}_{args.seed}_{int(time.time())}'
    args.run_name = run_name
    args.wandb_project_name = args.env + '_' + args.algo
    args.wandb_entity = 'jianzhnie'
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env, args.seed + i, i, args.capture_video, run_name)
        for i in range(args.num_processes)
    ])
    assert isinstance(
        envs.single_action_space,
        gym.spaces.Discrete), 'only discrete action space is supported'

    # get agent
    agent = Agent(
        args,
        envs=envs,
        algo=args.algo,
        total_steps=args.total_steps,
        gamma=args.gamma,
        epsilon=args.epsilon,
        learning_rate=args.learning_rate,
        update_target_step=args.update_target_step,
        device=device)
    agent.train()


if __name__ == '__main__':
    main()
