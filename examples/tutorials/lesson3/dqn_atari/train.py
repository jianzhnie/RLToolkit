import argparse
import sys

import gym
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../../../../')
from agent import Agent
from replay_memory import Experience, ReplayMemory

from rltoolkit.env.atari_wrappers import wrap_deepmind
from rltoolkit.utils import logger, tensorboard

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

config = {
    'train_seed': 42,
    'test_seed': 42,
    'env': 'PongNoFrameskip-v4',
    'algo': 'dqn',
    'use_wandb': False,
    'total_steps': 12000,  # max training steps
    'memory_size': 10000,  # Replay buffer size
    'memory_warmup_size': 5000,  # Replay buffer memory_warmup_size
    'batch_size': 32,  # repaly sample batch size
    'update_target_step': 100,  # target model update freq
    'learning_rate': 0.001,  # start learning rate
    'epsilon': 1,  # start greedy epsilon
    'epsilon_decay': 0.95,  # epsilon decay rate
    'min_epsilon': 0.1,
    'gamma': 0.99,  # discounting factor
    'image_size': (84, 84),
    'context_len': 4,
    'eval_render': True,  # do eval render
    'train_log_interval': 1,
    'test_log_interval': 5,  # evaluation freq
    'video_folder': 'results'
}


# train an episode
def run_train_episode(agent: Agent,
                      env: gym.Env,
                      rpm: ReplayMemory,
                      memory_warmup_size: int,
                      context_len: int = 4):
    total_reward = 0
    step = 0
    obs = env.reset()
    done = False
    while not done:
        step += 1
        context = rpm.recent_obs()
        context.append(obs)
        context = np.stack(context, axis=0)

        action = agent.sample(context)
        next_obs, reward, done, _ = env.step(action)
        rpm.append(Experience(obs, action, reward, done))

        # train model
        if rpm.size() > memory_warmup_size:
            # s,a,r,s',done
            (batch_all_obs, batch_action, batch_reward,
             batch_terminal) = rpm.sample_batch()
            batch_obs = batch_all_obs[:, :context_len, :, :]
            batch_next_obs = batch_all_obs[:, 1:, :, :]

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
        env.seed(np.random.randint(100))
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
                env.close()
        eval_rewards.append(score)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    return mean_reward, std_reward


def main(config):
    args = argparse.Namespace(**config)
    env = gym.make(args.env)
    env = wrap_deepmind(
        env, dim=args.image_size[0], framestack=False, obs_format='NCHW')
    test_env = gym.make(args.env)
    test_env = wrap_deepmind(
        test_env, dim=args.image_size[0], obs_format='NCHW', test=True)

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

    action_dim = env.action_space.n
    rpm = ReplayMemory(
        max_size=args.memory_size,
        batch_size=args.batch_size,
        obs_shape=args.image_size,
        context_len=args.context_len)

    # get agent
    agent = Agent(
        action_dim=action_dim,
        algo=args.algo,
        gamma=args.gamma,
        epsilon=args.epsilon,
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
    episode_cnt = 0
    cum_steps = 0  # this is the current timestep
    while cum_steps < args.total_steps:
        # start epoch
        episode_cnt += 1
        total_reward, steps = run_train_episode(
            agent, env, rpm, memory_warmup_size=args.memory_warmup_size)

        agent.epsilon *= args.epsilon_decay
        agent.epsilon = max(agent.epsilon, args.min_epsilon)

        cum_steps += steps
        pbar.update(steps)
        if episode_cnt % args.train_log_interval == 0:
            logger.info(
                '[Train], steps: {}, exploration:{}, learning rate:{}, Reward Sum {}.'
                .format(cum_steps, agent.epsilon,
                        agent.optimizer.param_groups[0]['lr'], total_reward))
            tensorboard.add_scalar('{}/training_rewards'.format(args.algo),
                                   total_reward, cum_steps)
            tensorboard.add_scalar('{}/exploration'.format(args.algo),
                                   agent.epsilon, cum_steps)

            if args.use_wandb:
                wandb.log({
                    'epsilon': agent.epsilon,
                    'train-score': total_reward
                })

        # perform evaluation
        if episode_cnt % args.test_log_interval == 0:
            mean_reward, std_reward = run_evaluate_episodes(agent, test_env)
            logger.info('[Eval], steps: {}, mean: {}, std: {:.2f}'.format(
                cum_steps, mean_reward, std_reward))
            tensorboard.add_scalar(
                '{}/mean_validation_rewards'.format(args.algo), mean_reward,
                cum_steps)

            if args.use_wandb:
                wandb.log({'test-score': mean_reward})

    pbar.close()
    # render and record video
    run_evaluate_episodes(
        agent,
        test_env,
        n_eval_episodes=1,
        render=True,
        video_folder=args.video_folder)


if __name__ == '__main__':
    main(config)
