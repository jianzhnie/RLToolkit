import argparse
import sys

import gym
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../../../../')
from agent import Agent

from rltoolkit.data.buffer.replaybuffer import ReplayBuffer
from rltoolkit.utils import logger, tensorboard

config = {
    'train_seed': 42,
    'test_seed': 42,
    'env': 'CartPole-v0',
    'algo': 'dqn',
    'hidden_dim': 128,
    'total_steps': 10000,  # max training steps
    'memory_size': 10000,  # Replay buffer size
    'memory_warmup_size': 1000,  # Replay buffer memory_warmup_size
    'batch_size': 64,  # repaly sample batch size
    'update_target_step': 100,  # target model update freq
    'learning_rate': 0.001,  # start learning rate
    'epsilon': 1,  # start greedy epsilon
    'epsilon_decay': 0.9995,  # epsilon decay rate
    'min_epsilon': 0.1,
    'gamma': 0.99,  # discounting factor
    'eval_render': True,  # do eval render
    'test_every_steps': 1000,  # evaluation freq
    'video_folder': 'results'
}


# train an episode
def run_train_episode(agent: Agent, env: gym.Env, rpm: ReplayBuffer,
                      memory_warmup_size: int):
    total_reward = 0
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

            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
        total_reward += reward
        obs = next_obs
    return total_reward, step


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
        score = 0
        while not done:
            action = agent.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            score += reward
            if render:
                env.render()
            if done:
                obs = env.close()
        eval_rewards.append(score)
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

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    rpm = ReplayBuffer(
        obs_dim=obs_dim, max_size=args.memory_size, batch_size=args.batch_size)

    # get agent
    agent = Agent(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        action_dim=action_dim,
        algo=args.algo,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        learning_rate=args.learning_rate,
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

        logger.info(
            '[Train], steps: {}, exploration:{}, learning rate:{}, Reward Sum {}.'
            .format(cum_steps, agent.epsilon,
                    agent.optimizer.param_groups[0]['lr'], total_reward))
        tensorboard.add_scalar('{}/training_rewards'.format(args.algo),
                               total_reward, cum_steps)
        tensorboard.add_scalar('{}/exploration'.format(args.algo),
                               agent.epsilon, cum_steps)

        pbar.update(steps)

        # perform evaluation
        if cum_steps // args.test_every_steps >= test_flag:
            while cum_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            mean_reward, std_reward = run_evaluate_episodes(agent, test_env)
            logger.info('[Eval], steps: {}, mean: {}, std: {:.2f}'.format(
                cum_steps, mean_reward, std_reward))
            tensorboard.add_scalar(
                '{}/mean_validation_rewards'.format(args.algo), mean_reward,
                cum_steps)

    # render and record video
    run_evaluate_episodes(
        agent,
        test_env,
        n_eval_episodes=1,
        render=True,
        video_folder=args.video_folder)
    pbar.close()


if __name__ == '__main__':
    main()
