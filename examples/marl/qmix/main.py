import argparse
import os.path as osp
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

sys.path.append('../../')

from rltoolkit.data.buffer.ma_replaybuffer import EpisodeData, ReplayBuffer
from rltoolkit.utils import tensorboard
from rltoolkit.utils.log import get_root_logger

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


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
        for _ in range(5):
            batch = rpm.sample_batch(config['batch_size'])
            loss, td_error = agent.learn(**batch)
            mean_loss.append(loss)
            mean_td_error.append(td_error)

    mean_loss = np.mean(mean_loss) if mean_loss else None
    mean_td_error = np.mean(mean_td_error) if mean_td_error else None
    return episode_reward, episode_step, is_win, mean_loss, mean_td_error


def run_evaluate_episode(env, agent, num_episodes=5):
    eval_is_win_buffer = []
    eval_reward_buffer = []
    eval_steps_buffer = []
    for _ in range(num_episodes):
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

    eval_reward = np.mean(eval_reward_buffer)
    eval_steps = np.mean(eval_steps_buffer)
    eval_win_rate = np.mean(eval_is_win_buffer)
    return eval_reward, eval_steps, eval_win_rate


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
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    if args.use_wandb:
        if has_wandb:
            wandb.init(
                project=args.scenario + '_' + args.algo,
                config=args,
                entity='jianzhnie',
                sync_tensorboard=True)
        else:
            logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                'Metrics not being logged to wandb, try `pip install wandb`')

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
        gamma=config['gamma'],
        learning_rate=config['lr'],
        exploration_start=config['exploration_start'],
        min_exploration=config['min_exploration'],
        exploration_decay=config['exploration_decay'],
        update_target_interval=config['update_target_interval'],
        clip_grad_norm=config['clip_grad_norm'],
        device=device)

    while rpm.size() < config['memory_warmup_size']:
        print('episode warmup')
        train_reward, train_step, train_is_win, train_loss, train_td_error = run_train_episode(
            env, qmix_agent, rpm, config)

    total_steps = 0
    episode_cnt = 0
    progress_bar = mmcv.ProgressBar(config['training_steps'])
    while total_steps < config['training_steps']:
        episode_cnt += 1
        train_reward, train_step, train_is_win, train_loss, train_td_error = run_train_episode(
            env, qmix_agent, rpm, config)
        total_steps += train_step

        tensorboard.add_scalar('train_rewards', train_reward, total_steps)
        tensorboard.add_scalar('train_episode_len', train_step, total_steps)
        tensorboard.add_scalar('train_win_rate', train_is_win, total_steps)
        tensorboard.add_scalar('train_loss', train_loss, total_steps)
        tensorboard.add_scalar('exploration', qmix_agent.exploration,
                               total_steps)
        tensorboard.add_scalar('replay_buffer_size', rpm.size(), total_steps)
        tensorboard.add_scalar('target_update_count',
                               qmix_agent.target_update_count, total_steps)
        tensorboard.add_scalar('train_td_error:', train_td_error, total_steps)

        if episode_cnt % config['train_log_interval'] == 0:
            logger.info(
                '[Train], total_steps: {}, episode_length: {}, train_win_rate: {:.2f}, train_reward: {:.2f}, train_loss: {:.2f}.'
                .format(total_steps, train_step, train_is_win, train_reward,
                        train_loss))

        if episode_cnt % config['test_interval'] == 0:
            eval_reward, eval_step, eval_is_win = run_evaluate_episode(
                env, qmix_agent)
            logger.info(
                '[Eval], eval_step: {}, eval_win_rate: {:.2f}, eval_reward: {:.2f}'
                .format(eval_step, eval_is_win, eval_reward))

            tensorboard.add_scalar('eval_reward', eval_reward, total_steps)
            tensorboard.add_scalar('eval_steps', eval_step, total_steps)
            tensorboard.add_scalar('eval_win_rate', eval_is_win, total_steps)

            if args.use_wandb:
                wandb.log({
                    'eval_reward': eval_reward,
                    'eval_steps': eval_step,
                    'eval_win_rate': eval_is_win
                })

        progress_bar.update()


if __name__ == '__main__':
    main()
