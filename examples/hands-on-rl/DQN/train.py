import argparse
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('../../../')
from agent import Agent

from rltoolkit.data.buffer.replaybuffer import ReplayBuffer
from rltoolkit.models.base_model import Model
from rltoolkit.policy.modelfree.dqn import DQN
from rltoolkit.utils import logger, tensorboard


class Qnet(Model):
    """只有一层隐藏层的Q网络."""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # 隐藏层使用ReLU激活函数
        return x


config = {
    'train_seed': 42,
    'env': 'CartPole-v0',
    'hidden_dim': 128,
    'total_steps': 100000,  # max training steps
    'memory_size': 50000,
    'memory_warmup_size': 10000,
    'update_freq': 50,
    'eval_episode': 100,
    'batch_size': 32,
    'update_target_step': 100,
    'start_lr': 0.0003,  # start learning rate
    'start_epslion': 1,
    'eps': 1e-5,  # Adam optimizer epsilon (default: 1e-5)
    'gamma': 0.99,  # discounting factor
    'eval_render': True,
    'test_every_steps': 10000,
    'video_folder': 'results'
}


# train an episode
def run_train_episode(agent: Agent, env: None, rpm: ReplayBuffer, args):
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
        if (rpm.size() > args.memory_warmup_size) and (step % args.update_freq
                                                       == 0):
            # s,a,r,s',done
            batchs = rpm.sample_batch(args.batch_size)
            train_loss = agent.learn(batchs)
            loss_lst.append(train_loss)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward, step, np.mean(loss_lst)


def run_evaluate_episodes(agent: Agent, env: None, args: None):
    env = gym.wrappers.RecordVideo(env, video_folder=args.video_folder)
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        score += reward
        if args.eval_render:
            env.render()
        if done:
            obs = env.reset()
    return score


def main():
    algo_name = 'dqn'
    args = argparse.Namespace(**config)
    env = gym.make(args.env)
    test_env = gym.make(args.env)
    # set seed
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    rpm = ReplayBuffer(obs_dim=state_dim, max_size=args.memory_size)
    # get model
    model = Qnet(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim)
    # get algorithm
    alg = DQN(model, gamma=args.gamma, lr=args.start_lr, device=device)
    # get agent
    agent = Agent(
        alg,
        act_dim=action_dim,
        total_step=args.total_steps,
        update_target_step=args.update_target_step,
        start_lr=args.start_lr,
        start_epslion=args.start_epslion,
        device=device)

    # start training, memory warm up
    with tqdm(
            total=args.memory_warmup_size,
            desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < args.memory_warmup_size:
            total_reward, steps, _ = run_train_episode(agent, env, rpm, args)
            pbar.update(steps)

    test_flag = 0
    train_total_steps = args.total_steps
    pbar = tqdm(total=train_total_steps)
    cum_steps = 0  # this is the current timestep
    while cum_steps < train_total_steps:
        # start epoch
        total_reward, steps, loss = run_train_episode(agent, env, rpm, args)
        cum_steps += steps

        pbar.set_description('[train]exploration, learning rate:{}, {}'.format(
            agent.curr_ep, agent.alg.optimizer.param_groups[0]['lr']))
        tensorboard.add_scalar('{}/training_rewards'.format(algo_name),
                               total_reward, cum_steps)
        tensorboard.add_scalar('{}/loss'.format(algo_name), loss,
                               cum_steps)  # mean of total loss
        tensorboard.add_scalar('{}/exploration'.format(algo_name),
                               agent.curr_ep, cum_steps)

        pbar.update(steps)

        # perform evaluation
        if cum_steps // args.test_every_steps >= test_flag:
            while cum_steps // args.test_every_steps >= test_flag:
                test_flag += 1

            pbar.write('testing')
            eval_rewards_mean = run_evaluate_episodes(agent, test_env, args)

            logger.info(
                'eval_agent done, (steps, eval_reward): ({}, {})'.format(
                    cum_steps, eval_rewards_mean))

            tensorboard.add_scalar(
                '{}/mean_validation_rewards'.format(algo_name),
                eval_rewards_mean, cum_steps)

    pbar.close()


if __name__ == '__main__':
    main()
