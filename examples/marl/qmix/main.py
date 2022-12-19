import argparse
import os
import sys
import time
from copy import deepcopy

import mmcv
import numpy as np
import torch
from env_wrapper import SC2EnvWrapper
from qmix_agent import QMixAgent
from qmix_config import QMixConfig
from qmixer_model import QMixerModel
from rnn_model import RNNModel
from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../')

from rltoolkit.data.buffer.ma_replaybuffer import EpisodeData, ReplayBuffer
from rltoolkit.utils import TensorboardLogger, WandbLogger
from rltoolkit.utils.logger.logs import get_outdir, get_root_logger


def run_train_episode(env: StarCraft2Env,
                      agent: QMixAgent,
                      rpm: ReplayBuffer,
                      config: dict = None):

    episode_limit = config['episode_limit']
    agent.reset_agent()
    episode_reward = 0.0
    episode_step = 0
    terminated = False
    state, obs = env.reset()
    episode_experience = EpisodeData(
        episode_limit=episode_limit,
        state_shape=config['state_shape'],
        obs_shape=config['obs_shape'],
        num_actions=config['n_actions'],
        num_agents=config['n_agents'],
    )

    while not terminated:
        available_actions = env.get_available_actions()
        actions = agent.sample(obs, available_actions)
        next_state, next_obs, reward, terminated = env.step(actions)
        episode_reward += reward
        episode_step += 1
        episode_experience.add(state, obs, actions, available_actions, reward,
                               terminated, 0)
        state = next_state
        obs = next_obs

    # fill the episode
    for _ in range(episode_step, episode_limit):
        episode_experience.fill_mask()

    episode_data = episode_experience.get_data()

    rpm.store(**episode_data)
    is_win = env.win_counted

    mean_loss = []
    mean_td_error = []
    if rpm.size() > config['memory_warmup_size']:
        for _ in range(config['update_learner_freq']):
            batch = rpm.sample_batch(config['batch_size'])
            loss, td_error = agent.learn(**batch)
            mean_loss.append(loss)
            mean_td_error.append(td_error)

    mean_loss = np.mean(mean_loss) if mean_loss else None
    mean_td_error = np.mean(mean_td_error) if mean_td_error else None

    return episode_reward, episode_step, is_win, mean_loss, mean_td_error


def run_evaluate_episode(env: StarCraft2Env,
                         agent: QMixAgent,
                         num_eval_episodes=5):
    eval_is_win_buffer = []
    eval_reward_buffer = []
    eval_steps_buffer = []
    for _ in range(num_eval_episodes):
        agent.reset_agent()
        episode_reward = 0.0
        episode_step = 0
        terminated = False
        state, obs = env.reset()
        while not terminated:
            available_actions = env.get_available_actions()
            actions = agent.predict(obs, available_actions)
            state, obs, reward, terminated = env.step(actions)
            episode_step += 1
            episode_reward += reward

        is_win = env.win_counted

        eval_reward_buffer.append(episode_reward)
        eval_steps_buffer.append(episode_step)
        eval_is_win_buffer.append(is_win)

    eval_rewards = np.mean(eval_reward_buffer)
    eval_steps = np.mean(eval_steps_buffer)
    eval_win_rate = np.mean(eval_is_win_buffer)

    return eval_rewards, eval_steps, eval_win_rate


def main():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = deepcopy(QMixConfig)
    env = StarCraft2Env(
        map_name=config['scenario'], difficulty=config['difficulty'])

    env = SC2EnvWrapper(env)
    config['episode_limit'] = env.episode_limit
    config['obs_shape'] = env.obs_shape
    config['state_shape'] = env.state_shape
    config['n_agents'] = env.n_agents
    config['n_actions'] = env.n_actions
    args = argparse.Namespace(**config)

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log
    log_name = os.path.join(args.project, args.scenario, args.algo, timestamp)
    text_log_path = os.path.join(args.log_dir, args.project, args.scenario,
                                 args.algo)
    tensorboard_log_path = get_outdir(text_log_path, 'log_dir')
    log_file = os.path.join(text_log_path, f'{timestamp}.log')
    text_logger = get_root_logger(log_file=log_file, log_level='INFO')

    if args.logger == 'wandb':
        logger = WandbLogger(
            train_interval=args.train_log_interval,
            test_interval=args.test_log_interval,
            update_interval=args.train_log_interval,
            project=args.project,
            name=log_name.replace(os.path.sep, '_'),
            save_interval=1,
            config=args,
            entity='jianzhnie')
    writer = SummaryWriter(tensorboard_log_path)
    writer.add_text('args', str(args))
    if args.logger == 'tensorboard':
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    rpm = ReplayBuffer(
        max_size=config['replay_buffer_size'],
        episode_limit=config['episode_limit'],
        state_shape=config['state_shape'],
        obs_shape=config['obs_shape'],
        num_agents=config['n_agents'],
        num_actions=config['n_actions'],
        batch_size=config['batch_size'],
        device=device)

    agent_model = RNNModel(
        input_shape=config['obs_shape'],
        n_actions=config['n_actions'],
        rnn_hidden_dim=config['rnn_hidden_dim'])
    qmixer_model = QMixerModel(
        n_agents=config['n_agents'],
        state_shape=config['state_shape'],
        mixing_embed_dim=config['mixing_embed_dim'],
        hypernet_layers=config['hypernet_layers'],
        hypernet_embed_dim=config['hypernet_embed_dim'])

    qmix_agent = QMixAgent(
        agent_model=agent_model,
        qmixer_model=qmixer_model,
        n_agents=config['n_agents'],
        double_q=config['double_q'],
        total_steps=config['total_steps'],
        gamma=config['gamma'],
        learning_rate=config['learning_rate'],
        exploration_start=config['exploration_start'],
        min_exploration=config['min_exploration'],
        update_target_interval=config['update_target_interval'],
        update_learner_freq=config['update_learner_freq'],
        device=device)

    progress_bar = mmcv.ProgressBar(config['memory_warmup_size'])
    while rpm.size() < config['memory_warmup_size']:
        train_reward, train_step, train_is_win, train_loss, train_td_error = run_train_episode(
            env, qmix_agent, rpm, config)
        progress_bar.update()

    current_steps = 0
    episode_cnt = 0
    progress_bar = mmcv.ProgressBar(config['total_steps'])
    while current_steps < config['total_steps']:
        episode_cnt += 1
        train_reward, train_step, train_is_win, train_loss, train_td_error = run_train_episode(
            env, qmix_agent, rpm, config)
        current_steps += train_step
        train_results = {
            'env_step': train_step,
            'rewards': train_reward,
            'win_rate': train_is_win,
            'mean_loss': train_loss,
            'mean_td_error': train_td_error,
            'exploration': qmix_agent.exploration,
            'replay_buffer_size': rpm.size(),
            'target_update_count': qmix_agent.target_update_count,
        }
        logger.log_train_data(train_results, current_steps)

        if episode_cnt % config['train_log_interval'] == 0:
            text_logger.info(
                '[Train], current_steps: {}, train_win_rate: {:.2f}, train_reward: {:.2f}'
                .format(current_steps, train_is_win, train_reward))

        if episode_cnt % config['test_log_interval'] == 0:
            eval_rewards, eval_steps, eval_win_rate = run_evaluate_episode(
                env, qmix_agent, num_eval_episodes=5)
            text_logger.info(
                '[Eval], eval_steps: {}, eval_win_rate: {:.2f}, eval_rewards: {:.2f}'
                .format(eval_steps, eval_win_rate, eval_rewards))

            test_results = {
                'env_step': eval_steps,
                'rewards': eval_rewards,
                'win_rate': eval_win_rate
            }
            logger.log_test_data(test_results, current_steps)

        progress_bar.update()


if __name__ == '__main__':
    main()
