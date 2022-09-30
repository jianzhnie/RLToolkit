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
    'env': 'ma_gym:Switch2-v1',
    'use_wandb': True,
    'algo': 'dqn',
    'hidden_dim': 128,
    'total_steps': 100000,  # max training steps
    'memory_size': 50000,  # Replay buffer size
    'memory_warmup_size': 10000,  # Replay buffer memory_warmup_size
    'batch_size': 64,  # repaly sample batch size
    'update_target_step': 100,  # target model update freq
    'learning_rate': 0.0005,  # start learning rate
    'epsilon': 1,  # start greedy epsilon
    'min_epsilon': 0.1,
    'gamma': 0.99,  # discounting factor
    'eval_render': True,  # do eval render
    'test_every_steps': 1000,  # evaluation freq
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
    return np.sum(score), step


def run_evaluate_episodes(agent: Agent,
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
        score = np.zeros(env.n_agents)
        while not done:
            action = agent.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            score += np.array(reward)
            if render:
                env.render()
            if done:
                obs = env.close()
        eval_rewards.append(np.sum(score))
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
        gamma=args.gamma,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        learning_rate=args.learning_rate,
        total_steps=args.total_steps,
        update_target_step=args.update_target_step,
        device=device)

    # start training, memory warm up
    with tqdm(
            total=args.memory_warmup_size,
            desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < args.memory_warmup_size:
            total_reward, steps = run_train_episode(
                agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
            pbar.update(steps)

    pbar = tqdm(total=args.total_steps)
    cum_steps = 0  # this is the current timestep
    test_flag = 0
    while cum_steps < args.total_steps:
        # start epoch
        total_reward, steps = run_train_episode(
            agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
        cum_steps += steps
        if args.use_wandb:
            wandb.log({
                'steps': cum_steps,
                'epsilon': agent.epsilon,
                'train-score': total_reward
            })

        pbar.update(steps)

        # perform evaluation
        if cum_steps // args.test_every_steps >= test_flag:
            while cum_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            mean_reward, std_reward = run_evaluate_episodes(agent, test_env)

        if args.use_wandb:
            wandb.log({
                'steps': cum_steps,
                'epsilon': agent.epsilon,
                'mean_reward': mean_reward,
                'std_reward': std_reward
            })
    pbar.close()
    # render and record video
    run_evaluate_episodes(agent, test_env, n_eval_episodes=1, render=True)


if __name__ == '__main__':
    import wandb
    wandb.init(project='minimal-marl', config=config, monitor_gym=True)
    main()
