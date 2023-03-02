import argparse
import sys

import gym
import numpy as np
import torch
from agent import Agent

sys.path.append('../../../../')
from rltoolkit.utils import logger

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

CartPole_config = {
    'train_seed': 42,
    'test_seed': 42,
    'env': 'CartPole-v1',
    'algo': 'reinforce',
    'use_wandb': True,
    'total_episode': 800,  # max training steps
    'hidden_dim': 128,
    'lr': 0.001,  # start learning rate
    'gamma': 0.99,  # discounting factor
    'with_baseline': False,
    'eval_render': False,  # do eval render
    'log_interval': 1,
    'test_every_episode': 10,  # evaluation freq
    'video_folder': 'results'
}


# 训练一个episode
def run_episode(env: gym.Env, agent: Agent):
    rewards, log_probs = [], []
    obs = env.reset()
    done = False
    while not done:
        action, log_prob = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        obs = next_obs
    return rewards, log_probs


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env: gym.Env,
             agent: Agent,
             n_eval_episodes: int = 5,
             render: bool = False,
             video_folder: str = None):
    """Evaluate the agent for ``n_eval_episodes`` episodes and returns average
    reward and std of reward.

    :param env: The evaluation environment
    :param agent : The evaluation agent
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param render: render the gym env
    """
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
    eval_rewards = []
    for i in range(n_eval_episodes):
        env.seed(np.random.randint(100))
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            score += reward
            if render:
                env.render()
            if done:
                env.close()
        eval_rewards.append(score)
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


def main(config):
    args = argparse.Namespace(**config)
    env = gym.make(args.env)
    test_env = gym.make(args.env)

    # set seed
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)
    env.seed(args.train_seed)
    test_env.seed(args.test_seed)

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

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    logger.info('obs_dim {}, action_dim {}'.format(obs_dim, action_dim))

    agent = Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        device=device)

    for curr_episode in range(args.total_episode):
        rewards, log_probs = run_episode(env, agent)
        episode_rewards = sum(rewards)
        returns = calc_discount_rewards(rewards, gamma=args.gamma)
        if args.with_baseline:
            loss = agent.learn_with_baseline(
                log_probs=log_probs, returns=returns)
        else:
            loss = agent.learn(log_probs=log_probs, returns=returns)

        if (curr_episode + 1) % args.log_interval == 0:
            logger.info('Episode {}, Loss {:.2f}, Reward Sum {}.'.format(
                curr_episode, loss, episode_rewards))

            if args.use_wandb:
                wandb.log({'train-score': episode_rewards})

        if (curr_episode + 1) % args.test_every_episode == 0:
            mean_reward, std_reward = evaluate(
                env, agent, n_eval_episodes=5, render=args.eval_render)
            logger.info('Test reward: mean: {}, std: {:.2f}'.format(
                mean_reward, std_reward))

            if args.use_wandb:
                wandb.log({'test-score': mean_reward})

    mean_reward, std_reward = evaluate(
        test_env,
        agent,
        n_eval_episodes=1,
        render=True,
        video_folder=args.video_folder)


if __name__ == '__main__':
    main(CartPole_config)
