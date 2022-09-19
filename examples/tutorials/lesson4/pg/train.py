import argparse
import sys

import gym
import numpy as np
import torch
from agent import Agent

sys.path.append('../../../../')
from rltoolkit.utils import logger


# 训练一个episode
def run_episode(env, agent):
    obs_list, action_list, log_prob_list, reward_list = [], [], [], []
    obs = env.reset()
    while True:
        action, log_prob = agent.sample(obs)
        obs, reward, done, _ = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        log_prob_list.append(log_prob)
        reward_list.append(reward)
        if done:
            break
    return obs_list, action_list, log_prob_list, reward_list


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, n_episode=5, render=False):
    eval_reward = []
    for i in range(n_episode):
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
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def calc_discount_rewards(rewards, gamma):
    G = 0
    returns = []
    for i in reversed(range(len(rewards))):
        reward = rewards[i]
        # G_i = r_i + γ·G_i+1
        G = gamma * G + reward
        returns.insert(0, G)
    return returns


config = {
    'train_seed': 42,
    'test_seed': 42,
    'env': 'CartPole-v0',
    'hidden_dim': 128,
    'total_episode': 800,  # max training steps
    'batch_size': 64,  # repaly sample batch size
    'start_lr': 0.001,  # start learning rate
    'end_lr': 0.00001,  # end learning rate
    'gamma': 0.98,  # discounting factor
    'eval_render': False,  # do eval render
    'test_every_episode': 50,  # evaluation freq
    'video_folder': 'results'
}


def main():
    args = argparse.Namespace(**config)
    env = gym.make(args.env)
    test_env = gym.make(args.env)

    # set seed
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)
    env.seed(args.train_seed)
    test_env.seed(args.test_seed)
    # env = env.unwrapped # Cancel the minimum score limit
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    logger.info('obs_dim {}, action_dim {}'.format(obs_dim, action_dim))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    agent = Agent(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.start_lr,
        gamma=args.gamma,
        device=device)

    for i in range(args.total_episode):
        obs_list, action_list, log_prob_list, reward_list = run_episode(
            env, agent)
        logger.info('Episode {}, Reward Sum {}.'.format(i, sum(reward_list)))
        return_list = calc_discount_rewards(reward_list, gamma=args.gamma)
        # agent.update(
        #     obs_list=obs_list,
        #     action_list=action_list,
        #     reward_list=reward_list)
        agent.updat_with_baseline(log_probs=log_prob_list, returns=return_list)
        if (i + 1) % args.test_every_episode == 0:
            total_reward = evaluate(env, agent, render=args.eval_render)
            logger.info('Test reward: {}'.format(total_reward))


if __name__ == '__main__':
    main()
