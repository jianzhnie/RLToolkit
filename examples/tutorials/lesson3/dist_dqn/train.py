import argparse
import sys

import gym
import torch
from tqdm import tqdm

sys.path.append('../../../../')
import time

from agent import Agent

from rltoolkit.data.buffer.replaybuffer import ReplayBuffer
from rltoolkit.utils import logger, tensorboard
from rltoolkit.utils.utils import make_env

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

config = {
    'seed': 42,
    'env': 'CartPole-v0',
    'num_processes': 1,
    'algo': 'dqn',
    'use_wandb': False,
    'capture_video': True,
    'hidden_dim': 128,
    'total_steps': 10000,  # max training steps
    'memory_size': 10000,  # Replay buffer size
    'memory_warmup_size': 1000,  # Replay buffer memory_warmup_size
    'batch_size': 32,  # repaly sample batch size
    'update_target_step': 100,  # target model update freq
    'learning_rate': 0.001,  # start learning rate
    'epsilon': 1,  # start greedy epsilon
    'epsilon_decay': 0.97,  # epsilon decay rate
    'min_epsilon': 0.1,
    'gamma': 0.99,  # discounting factor
    'eval_render': True,  # do eval render
    'train_log_interval': 1,
    'test_log_interval': 5,  # evaluation freq
    'log_dir': 'work_dir',
    'video_folder': 'results'
}


# train an episode
def run_train_episode(args, agent: Agent, env: gym.Env, rpm: ReplayBuffer):
    total_reward = 0
    step = 0
    obs = env.reset()
    done = False
    while not done:
        step += 1
        actions = agent.sample(obs)
        next_obs, rewards, dones, infos = env.step(actions)

        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]['terminal_observation']

        rpm.add(obs, real_next_obs, actions, rewards, dones, infos)
        # train model
        if rpm.size() > args.memory_warmup_size:
            # s,a,r,s',done
            samples = rpm.sample(args.batch_size)

            batch_obs = samples.obs
            batch_action = samples.actions
            batch_reward = samples.rewards
            batch_next_obs = samples.next_obs
            batch_terminal = samples.dones

            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
        total_reward += rewards
        obs = next_obs
    return total_reward, step


def main():
    args = argparse.Namespace(**config)
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    run_name = f'{args.env}_{args.algo}_{args.seed}_{int(time.time())}'
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env, args.seed, 0, args.capture_video, run_name)
        for i in range(args.num_processes)
    ])
    assert isinstance(
        envs.single_action_space,
        gym.spaces.Discrete), 'only discrete action space is supported'

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

    rpm = ReplayBuffer(
        buffer_size=args.memory_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        n_envs=envs.num_envs,
        device=device,
        handle_timeout_termination=True)

    # get agent
    agent = Agent(
        env=envs,
        hidden_dim=args.hidden_dim,
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
            total_reward, steps = run_train_episode(args, agent, envs, rpm)
            pbar.update(steps)

    pbar = tqdm(total=args.total_steps)
    episode_cnt = 0
    cum_steps = 0  # this is the current timestep
    while cum_steps < args.total_steps:
        # start epoch
        episode_cnt += 1
        total_reward, steps = run_train_episode(args, agent, envs, rpm)

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

    pbar.close()


if __name__ == '__main__':
    main()
