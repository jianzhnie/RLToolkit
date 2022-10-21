import argparse
import sys

import gym
import numpy as np
import torch
from ddpg_agent import Agent
from tqdm import tqdm

from rltoolkit.data.buffer.replaybuffer import ReplayBuffer

sys.path.append('../../../../')
from rltoolkit.utils import logger, tensorboard

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

config = {
    'train_seed': 42,
    'test_seed': 42,
    'env': 'BipedalWalker-v3',
    'algo': 'ddpgn',
    'use_wandb': True,
    'hidden_dim': 128,
    'total_steps': 1000000,  # max training steps
    'memory_size': 200000,  # Replay buffer size
    'memory_warmup_size': 10000,  # Replay buffer memory_warmup_size
    'actor_lr': 3e-4,  # start learning rate
    'critic_lr': 3e-3,  # end learning rate
    'initial_random_steps': 2000,
    'ou_noise_theta': 1.0,
    'ou_noise_sigma': 0.1,
    'weight_decay': 0.0001,
    'gamma': 0.99,  # discounting factor
    'tau': 0.005,  # 软更新参数,
    'sigma': 0.01,
    'batch_size': 128,
    'eval_render': False,  # do eval render
    'train_log_interval': 1,
    'test_log_interval': 5,  # evaluation freq
    'video_folder': 'results'
}


# train an episode
def run_train_episode(agent: Agent, env: gym.Env, rpm: ReplayBuffer,
                      memory_warmup_size: int):
    total_reward = 0
    policy_loss_lst = []
    value_loss_lst = []
    step = 0
    obs = env.reset()
    done = False
    while not done:
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)
        # train model
        if rpm.size() > memory_warmup_size:
            # s,a,r,s',done
            samples = rpm.sample_batch()

            batch_obs = samples['obs']
            batch_action = samples['action']
            batch_reward = samples['reward']
            batch_next_obs = samples['next_obs']
            batch_terminal = samples['terminal']

            policy_loss, value_loss = agent.learn(batch_obs, batch_action,
                                                  batch_reward, batch_next_obs,
                                                  batch_terminal)
            policy_loss_lst.append(policy_loss)
            value_loss_lst.append(value_loss)

        total_reward += reward
        obs = next_obs
    return total_reward, step, np.mean(policy_loss_lst), np.mean(
        value_loss_lst)


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(agent: Agent,
             env: gym.Env,
             n_eval_episodes: int = 5,
             render: bool = False,
             video_folder: str = None):

    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
    eval_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                env.close()
        eval_rewards.append(episode_reward)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    return mean_reward, std_reward


def main():
    args = argparse.Namespace(**config)
    env = gym.make(args.env)
    test_env = gym.make(args.env)

    # set seed
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)
    env.seed(args.train_seed)
    test_env.seed(args.test_seed)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.use_wandb:
        if has_wandb:
            wandb.init(
                project=args.env + '_' + args.algo,
                config=args,
                entity='jianzhnie',
                sync_tensorboard=True,
                monitor_gym=True)
        else:
            logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                'Metrics not being logged to wandb, try `pip install wandb`')

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值

    rpm = ReplayBuffer(
        max_size=args.memory_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        batch_size=args.batch_size,
        device=device)

    agent = Agent(
        env=env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        initial_random_steps=args.initial_random_steps,
        action_bound=action_bound,
        tau=args.tau,
        gamma=args.gamma,
        device=device)

    # start training, memory warm up
    with tqdm(
            total=args.memory_warmup_size,
            desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < args.memory_warmup_size:
            total_reward, steps, _, _ = run_train_episode(
                agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
            pbar.update(steps)

    pbar = tqdm(total=args.total_steps, desc='[Training]')
    cum_steps = 0  # this is the current timestep
    episode_cnt = 0
    while cum_steps < args.total_steps:
        # start epoch
        episode_cnt += 1
        total_reward, steps, policy_loss, value_loss = run_train_episode(
            agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
        cum_steps += steps

        if episode_cnt % args.train_log_interval == 0:
            logger.info(
                'Current Steps: {}, Plicy Loss {:.2f}, Value Loss {:.2f}, Reward Sum {}.'
                .format(cum_steps, policy_loss, value_loss, total_reward))
            tensorboard.add_scalar('{}/training_rewards'.format(args.algo),
                                   total_reward, cum_steps)
            tensorboard.add_scalar('{}/policy_loss'.format(args.algo),
                                   policy_loss, cum_steps)
            tensorboard.add_scalar('{}/value_loss'.format(args.algo),
                                   value_loss, cum_steps)

            if args.use_wandb:
                wandb.log({'episode_length': steps})
                wandb.log({'policy_loss': policy_loss})
                wandb.log({'value_loss': value_loss})
                wandb.log({'train_reward': total_reward})

        pbar.update(steps)
        # perform evaluation
        if episode_cnt % args.test_log_interval == 0:
            mean_reward, std_reward = evaluate(agent, test_env)
            logger.info(
                'Eval_agent done, steps: {}, mean: {}, std: {:.2f}'.format(
                    cum_steps, mean_reward, std_reward))
            tensorboard.add_scalar(
                '{}/mean_validation_rewards'.format(args.algo), mean_reward,
                cum_steps)

            if args.use_wandb:
                wandb.log({'test-reward': mean_reward})
                wandb.log({'std_reward': std_reward})

    pbar.close()
    # render and record video
    mean_reward, std_reward = evaluate(
        agent,
        test_env,
        n_eval_episodes=1,
        render=False,
        video_folder=args.video_folder)


if __name__ == '__main__':
    main()
