import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('../../')
from rltoolkit.utils import rl_utils


class PolicyNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, dropout=0.6):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class REINFORCE(object):

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)
        # 使用Adam优化器
        self.gamma = gamma
        # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)
        return action.item(), log_probs

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降

    def update_(self, transition_dict):
        reward_list = transition_dict['rewards']
        log_prob_list = transition_dict['log_probs']

        G = 0
        returns = []
        policy_loss = []
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            G = self.gamma * G + reward
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, G in zip(log_prob_list, returns):
            policy_loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降

    def update_with_baseline(self, transition_dict):
        reward_list = transition_dict['rewards']
        log_prob_list = transition_dict['log_probs']

        G = 0
        returns = []
        policy_loss = []
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            G = self.gamma * G + reward
            returns.insert(0, G)

        returns = torch.tensor(returns)
        baseline = torch.mean(returns)
        for log_prob, G in zip(log_prob_list, returns):
            policy_loss.append(-log_prob * (G - baseline))

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降


if __name__ == '__main__':
    learning_rate = 1e-3
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.98
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    eps = np.finfo(np.float32).eps.item()
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                      device)

    return_list = []
    for i in range(10):
        with tqdm(int(num_episodes), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'log_probs': [],
                    'dones': []
                }
                state = env.reset()
                done = False
                while not done:
                    action, log_probs = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['log_probs'].append(log_probs)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update_with_baseline(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()
