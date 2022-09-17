import argparse
import sys

import gym
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../../../')
from agent import Agent

from rltoolkit.data.buffer.replaybuffer import ReplayBuffer
from rltoolkit.models.noisynet import NoisyNet
from rltoolkit.policy.modelfree import DDQN, DQN
from rltoolkit.utils import logger, tensorboard

config = {
    'train_seed': 42,
    'test_seed': 42,
    'env': 'CartPole-v0',
    'hidden_dim': 128,
    'total_steps': 10000,  # max training steps
    'memory_size': 2000,  # Replay buffer size
    'memory_warmup_size': 500,  # Replay buffer memory_warmup_size
    'batch_size': 64,  # repaly sample batch size
    'update_target_step': 100,  # target model update freq
    'start_lr': 0.01,  # start learning rate
    'end_lr': 0.00001,  # end learning rate
    'gamma': 0.99,  # discounting factor
    'eval_render': True,  # do eval render
    'test_every_steps': 1000,  # evaluation freq
    'video_folder': 'results'
}


# train an episode
def run_train_episode(agent: Agent, env: gym.Env, rpm: ReplayBuffer,
                      memory_warmup_size: int):
    total_reward = 0
    obs = env.reset()
    step = 0
    loss_lst = []
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

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_terminal)
            loss_lst.append(train_loss)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward, step, np.mean(loss_lst)


def run_evaluate_episodes(agent: Agent,
                          env: gym.Env,
                          render: bool = False,
                          video_folder: str = None):
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
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
    return score


def main():
    algo_name = 'dqn'
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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    rpm = ReplayBuffer(
        obs_dim=state_dim,
        max_size=args.memory_size,
        batch_size=args.batch_size)
    # get model
    model = NoisyNet(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim)
    # get algorithm
    if algo_name == 'dqn':
        alg = DQN(model, gamma=args.gamma, lr=args.start_lr, device=device)
    elif algo_name == 'ddqn':
        alg = DDQN(model, gamma=args.gamma, lr=args.start_lr, device=device)
    # get agent
    agent = Agent(
        alg,
        act_dim=action_dim,
        total_step=args.total_steps,
        update_target_step=args.update_target_step,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        device=device)

    # start training, memory warm up
    with tqdm(
            total=args.memory_warmup_size,
            desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < args.memory_warmup_size:
            total_reward, steps, _ = run_train_episode(
                agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
            pbar.update(steps)

    pbar = tqdm(total=args.total_steps, desc='[Training]')
    cum_steps = 0  # this is the current timestep
    test_flag = 0
    while cum_steps < args.total_steps:
        # start epoch
        total_reward, steps, loss = run_train_episode(
            agent, env, rpm, memory_warmup_size=args.memory_warmup_size)
        cum_steps += steps
        pbar.set_description('[train]learning rate:{}'.format(
            agent.alg.optimizer.param_groups[0]['lr']))
        tensorboard.add_scalar('{}/training_rewards'.format(algo_name),
                               total_reward, cum_steps)
        tensorboard.add_scalar('{}/loss'.format(algo_name), loss,
                               cum_steps)  # mean of total loss

        pbar.update(steps)
        # perform evaluation
        if cum_steps // args.test_every_steps >= test_flag:
            while cum_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            eval_rewards_mean = run_evaluate_episodes(agent, test_env)
            pbar.write('testing')
            logger.info(
                'eval_agent done, (steps, eval_reward): ({}, {})'.format(
                    cum_steps, eval_rewards_mean))

            tensorboard.add_scalar(
                '{}/mean_validation_rewards'.format(algo_name),
                eval_rewards_mean, cum_steps)

    # render and record video
    run_evaluate_episodes(
        agent, test_env, render=True, video_folder=args.video_folder)
    pbar.close()


if __name__ == '__main__':
    main()
