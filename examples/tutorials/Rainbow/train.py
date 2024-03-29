import argparse
import os
import time

import gym
import numpy as np
import torch
from agent import Agent
from torch.utils.tensorboard import SummaryWriter

from rltoolkit.data.buffer import MultiStepReplayBuffer
from rltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                             get_outdir, get_root_logger)

config = {
    'train_seed': 42,
    'test_seed': 42,
    'project': 'Classic-Control',
    'env': 'CartPole-v0',
    'algo': 'rainbow',
    'model_name': 'noisy-network',
    'hidden_dim': 128,
    'total_steps': 12000,  # max training steps
    'memory_size': 10000,  # Replay buffer size
    'memory_warmup_size': 500,  # Replay buffer memory_warmup_size
    'batch_size': 64,  # repaly sample batch size
    'update_target_step': 100,  # target model update freq
    'learning_rate': 0.001,  # start learning rate
    'end_lr': 0.00001,  # end learning rate
    'gamma': 0.99,  # discounting factor
    'v_min': 0.0,
    'v_max': 200.0,
    'atom_size': 51,
    'std_init': 0.5,
    'n_step': 5,
    'alpha': 0.2,
    'beta': 0.6,
    'prior_eps': 1e-6,
    'train_log_interval': 1,
    'test_log_interval': 5,  # evaluation freq
    'eval_render': True,  # do eval render
    'video_folder': 'results',
    'log_dir': 'work_dirs',
    'logger': 'wandb'
}


# train an episode
def run_train_episode(agent: Agent, env: gym.Env, rpm: MultiStepReplayBuffer,
                      memory_warmup_size: int):
    episode_reward = 0
    episode_step = 0
    episode_loss = []
    obs = env.reset()
    done = False
    while not done:
        episode_step += 1
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

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_terminal)
            episode_loss.append(train_loss)
        episode_reward += reward
        obs = next_obs
    return episode_reward, episode_step, np.mean(episode_loss)


def run_evaluate_episodes(agent: Agent,
                          env: gym.Env,
                          n_eval_episodes: int = 5,
                          render: bool = False,
                          video_folder: str = None):
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
    eval_rewards = []
    eval_steps = []
    for _ in range(n_eval_episodes):
        env.seed(np.random.randint(100))
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_step = 0
        while not done:
            action = agent.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            episode_reward += reward
            episode_step += 1
            if render:
                env.render()
            if done:
                env.close()
        eval_rewards.append(episode_reward)
        eval_steps.append(episode_step)
    mean_reward = np.mean(eval_rewards)
    mean_steps = np.mean(eval_steps)
    return mean_reward, mean_steps


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

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log
    log_name = os.path.join(args.project, args.env, args.algo, timestamp)
    text_log_path = os.path.join(args.log_dir, args.project, args.env,
                                 args.algo)
    tensorboard_log_path = get_outdir(text_log_path, 'log_dir')
    log_file = os.path.join(text_log_path, f'{timestamp}.log')
    text_logger = get_root_logger(log_file=log_file, log_level='INFO')
    args.video_folder = get_outdir(text_log_path, 'video')

    if args.logger == 'wandb':
        logger = WandbLogger(train_interval=args.train_log_interval,
                             test_interval=args.test_log_interval,
                             update_interval=args.train_log_interval,
                             project=args.project,
                             name=log_name.replace(os.path.sep, '_'),
                             config=args,
                             entity='jianzhnie')
    writer = SummaryWriter(tensorboard_log_path)
    writer.add_text('args', str(args))
    if args.logger == 'tensorboard':
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    rpm = MultiStepReplayBuffer(max_size=args.memory_size,
                                obs_dim=obs_dim,
                                batch_size=args.batch_size,
                                n_step=args.n_step,
                                gamma=args.gamma,
                                device=device)
    # get agent
    agent = Agent(model_name=args.model_name,
                  obs_dim=obs_dim,
                  action_dim=action_dim,
                  hidden_dim=args.hidden_dim,
                  total_steps=args.total_steps,
                  update_target_step=args.update_target_step,
                  learning_rate=args.learning_rate,
                  end_lr=args.end_lr,
                  std_init=args.std_init,
                  gamma=args.gamma,
                  batch_size=args.batch_size,
                  v_min=args.v_min,
                  v_max=args.v_max,
                  atom_size=args.atom_size,
                  n_step=args.n_step,
                  device=device)

    # start training, memory warm up
    progress_bar = ProgressBar(config['memory_warmup_size'])
    while rpm.size() < args.memory_warmup_size:
        episode_reward, episode_step, loss = run_train_episode(
            agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
        progress_bar.update(episode_step)

    episode_cnt = 0
    steps_cnt = 0  # this is the current timestep
    progress_bar = ProgressBar(args.total_steps)
    while steps_cnt < args.total_steps:
        # start epoch
        episode_reward, episode_step, episode_loss = run_train_episode(
            agent, env, rpm, memory_warmup_size=args.memory_warmup_size)

        steps_cnt += episode_step
        episode_cnt += 1

        train_results = {
            'env_step': episode_step,
            'rewards': episode_reward,
            'episode_loss': episode_loss,
            'learning_rate': agent.learning_rate,
            'replay_buffer_size': rpm.size()
        }

        if episode_cnt % config['train_log_interval'] == 0:
            text_logger.info(
                '[Train], episode: {},  train_reward: {:.2f}'.format(
                    episode_cnt, episode_reward))
            logger.log_train_data(train_results, steps_cnt)

        # perform evaluation
        if episode_cnt % config['test_log_interval'] == 0:
            eval_rewards, eval_steps = run_evaluate_episodes(agent,
                                                             test_env,
                                                             n_eval_episodes=5,
                                                             render=False)
            text_logger.info(
                '[Eval], episode: {},  eval_rewards: {:.2f}'.format(
                    episode_cnt, eval_rewards))

            test_results = {'env_step': eval_steps, 'rewards': eval_rewards}
            logger.log_test_data(test_results, steps_cnt)

        progress_bar.update(episode_step)

    # render and record video
    run_evaluate_episodes(agent,
                          test_env,
                          n_eval_episodes=1,
                          render=False,
                          video_folder=args.video_folder)


if __name__ == '__main__':
    main()
