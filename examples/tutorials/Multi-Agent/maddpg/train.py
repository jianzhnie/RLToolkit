import argparse
import sys

import gym
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../../../../')
from agent import Agent

from rltoolkit.data.buffer.ma_replaybuffer import ReplayBuffer

config = {
    'train_seed': 42,
    'test_seed': 42,
    'env': 'ma_gym:Switch2-v2',
    'use_wandb': True,
    'algo': 'dqn',
    'total_steps': 1000000,  # max training steps
    'memory_size': 100000,  # Replay buffer size
    'initial_random_steps': 5000,
    'memory_warmup_size': 2000,  # Replay buffer memory_warmup_size
    'batch_size': 32,  # repaly sample batch size
    'log_interval': 10,
    'actor_lr': 5e-4,  # start learning rate
    'critic_lr': 1e-3,  # end learning rate
    'gamma': 0.99,  # discounting factor
    'tau': 0.005,
    'max_episodes': 10000,
    'temperature': 0.1,
    'gumbel_max': 10,
    'gumbel_min': 0.1,
    'eval_render': True,  # do eval render
    'video_folder': 'results'
}


# train an episode
def run_train_episode(agent: Agent, env: gym.Env, rpm: ReplayBuffer,
                      memory_warmup_size: int):

    score = np.zeros(env.n_agents)
    step = 0
    obs = env.reset()
    done = [False for _ in range(env.n_agents)]
    while not all(done):
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.store(obs, action, reward, next_obs, done)
        # train model
        if rpm.size() > memory_warmup_size:
            # s,a,r,s',done
            samples = rpm.sample_batch()

            batch_obs = samples['obs']
            batch_action = samples['action']
            batch_reward = samples['reward']
            batch_next_obs = samples['next_obs']
            batch_terminal = samples['terminal']

            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
        score += np.array(reward)
        obs = next_obs
    return sum(score), step


def run_evaluate_episodes(agent: Agent,
                          env: gym.Env,
                          n_eval_episodes: int = 5,
                          render: bool = False,
                          video_folder: str = None):
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)

    score = np.zeros(env.n_agents)
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = [False for _ in range(env.n_agents)]
        while not all(done):
            action = agent.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            score += np.array(reward)
            if render:
                env.render()
    return sum(score / n_eval_episodes)


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

    obs_dim = env.observation_space[0].shape[0]
    num_agents = len(env.observation_space)

    rpm = ReplayBuffer(
        obs_dim=obs_dim,
        num_agents=num_agents,
        max_size=args.memory_size,
        batch_size=args.batch_size)

    # get agent
    agent = Agent(
        env=env,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        temperature=args.temperature,
        total_steps=args.total_steps,
        initial_random_steps=args.initial_random_steps,
        device=device)

    # start training, memory warm up
    with tqdm(
            total=args.memory_warmup_size,
            desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < args.memory_warmup_size:
            rewards, steps = run_train_episode(
                agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
            pbar.update(steps)

    pbar = tqdm(total=args.total_steps)
    episode_cnt = 0
    cum_steps = 0  # this is the current timestep
    episode_score = 0
    while cum_steps < args.total_steps:
        # start epoch
        episode_cnt += 1
        agent.temperature = max(
            args.gumbel_min,
            args.gumbel_max - (args.gumbel_max - args.gumbel_min) *
            (episode_cnt / (0.6 * args.max_episodes)))

        rewards, steps = run_train_episode(
            agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
        episode_score += rewards
        cum_steps += steps
        pbar.update(steps)
        if episode_cnt % args.log_interval == 0:
            train_score = episode_score / args.log_interval
            test_score = run_evaluate_episodes(agent, test_env)
            if args.use_wandb:
                wandb.log({
                    'episode': episode_cnt,
                    'test-score': test_score,
                    'train-score': train_score
                })
            episode_score = 0

    pbar.close()
    # render and record video
    run_evaluate_episodes(agent, test_env, n_eval_episodes=1, render=True)


if __name__ == '__main__':
    import wandb
    wandb.init(project='minimal-marl', config=config, monitor_gym=True)
    main()
