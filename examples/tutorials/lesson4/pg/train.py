import argparse
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import Agent

sys.path.append('../../../../')
from rltoolkit.utils import logger, rl_utils


# 训练一个episode
def run_episode(env: gym.Env, agent: Agent):
    rewards, log_probs = [], []
    obs = env.reset()
    while True:
        action, log_prob = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        obs = next_obs
        if done:
            break
    return rewards, log_probs


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env: gym.Env,
             agent: Agent,
             n_eval_episodes: int = 5,
             render: bool = False):
    eval_rewards = []
    for i in range(n_eval_episodes):
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
    'total_episode': 800,  # max training steps
    'hidden_dim': 128,
    'lr': 0.001,  # start learning rate
    'gamma': 0.98,  # discounting factor
    'with_baseline': False,
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
        learning_rate=args.lr,
        gamma=args.gamma,
        device=device)

    return_list = []
    for i_episode in range(args.total_episode):
        rewards, log_probs = run_episode(env, agent)
        episode_return = sum(rewards)
        returns = calc_discount_rewards(rewards, gamma=args.gamma)
        if args.with_baseline:
            loss = agent.learn_with_baseline(
                log_probs=log_probs, returns=returns)
        else:
            loss = agent.learn(log_probs=log_probs, returns=returns)
        if (i_episode + 1) % args.test_every_episode == 0:
            logger.info('Episode {}, Loss {:.2f}, Reward Sum {}.'.format(
                i_episode, loss, episode_return))
            mean_reward, std_reward = evaluate(
                env, agent, n_eval_episodes=5, render=args.eval_render)
            logger.info('Test reward: mean: {}, std: {:.2f}'.format(
                mean_reward, std_reward))
        return_list.append(episode_return)

    mean_reward, std_reward = evaluate(
        test_env, agent, n_eval_episodes=1, render=True)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(args.env))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(args.env))
    plt.show()


if __name__ == '__main__':
    main()
