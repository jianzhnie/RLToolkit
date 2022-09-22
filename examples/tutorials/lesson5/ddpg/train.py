import argparse
import sys

import gym
import numpy as np
import torch
from agent import Agent
from tqdm import tqdm

sys.path.append('../../../../')
from rltoolkit.data.buffer.replaybuffer import ReplayBuffer
from rltoolkit.utils import logger, tensorboard

config = {
    'train_seed': 42,
    'test_seed': 42,
    'env': 'Pendulum-v1',
    'total_episode': 800,  # max training steps
    'hidden_dim': 128,
    'total_steps': 10000,  # max training steps
    'memory_size': 5000,  # Replay buffer size
    'memory_warmup_size': 1000,  # Replay buffer memory_warmup_size
    'actor_lr': 3e-4,  # start learning rate
    'critic_lr': 3e-3,  # end learning rate
    'initial_random_steps': 2000,
    'ou_noise_theta': 1.0,
    'ou_noise_sigma': 0.1,
    'gamma': 0.98,  # discounting factor
    'tau': 0.005,  # 软更新参数,
    'sigma': 0.01,
    'batch_size': 64,
    'eval_render': False,  # do eval render
    'test_every_steps': 1000,  # evaluation freq
    'video_folder': 'results'
}


# train an episode
def run_train_episode(agent: Agent, env: gym.Env, rpm: ReplayBuffer,
                      memory_warmup_size: int):
    total_reward = 0
    obs = env.reset()
    step = 0
    policy_loss_lst = []
    value_loss_lst = []
    while True:
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
        if done:
            break
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
    algo_name = 'ddpg'
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
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    rpm = ReplayBuffer(
        obs_dim=obs_dim, max_size=args.memory_size, batch_size=args.batch_size)
    agent = Agent(
        env=env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        initial_random_steps=args.initial_random_steps,
        ou_noise_theta=args.ou_noise_theta,
        ou_noise_sigma=args.ou_noise_sigma,
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
    test_flag = 0
    while cum_steps < args.total_steps:
        # start epoch
        total_reward, steps, policy_loss, value_loss = run_train_episode(
            agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
        cum_steps += steps

        logger.info(
            'Current Steps: {}, Plicy Loss {:.2f}, Value Loss {:.2f}, Reward Sum {}.'
            .format(cum_steps, policy_loss, value_loss, total_reward))
        tensorboard.add_scalar('{}/training_rewards'.format(algo_name),
                               total_reward, cum_steps)
        tensorboard.add_scalar('{}/policy_loss'.format(algo_name), policy_loss,
                               cum_steps)
        tensorboard.add_scalar('{}/value_loss'.format(algo_name), value_loss,
                               cum_steps)

        pbar.update(steps)
        # perform evaluation
        if cum_steps // args.test_every_steps >= test_flag:
            while cum_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            mean_reward, std_reward = evaluate(agent, test_env)
            logger.info(
                'Eval_agent done, steps: {}, mean: {}, std: {:.2f}'.format(
                    cum_steps, mean_reward, std_reward))
            tensorboard.add_scalar(
                '{}/mean_validation_rewards'.format(algo_name), mean_reward,
                cum_steps)

    # render and record video
    mean_reward, std_reward = evaluate(
        agent,
        test_env,
        n_eval_episodes=1,
        render=True,
        video_folder=args.video_folder)
    pbar.close()


if __name__ == '__main__':
    main()
