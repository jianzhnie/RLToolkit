import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('../../')
from behavior_cloning import PPO, sample_expert_data

from rltoolkit.utils import rl_utils


class Discriminator(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


class GAIL:

    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.device = device

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(
            expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a).to(self.device)
        expert_actions = F.one_hot(expert_actions, num_classes=2).float()
        agent_actions = F.one_hot(agent_actions, num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        self.agent.update(transition_dict)


if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 250
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    lr_d = 1e-3
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d, device)
    n_episode = 500
    return_list = []

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    n_episode = 1
    expert_s, expert_a = sample_expert_data(n_episode)

    with tqdm(total=n_episode, desc='进度条') as pbar:
        for i in range(n_episode):
            episode_return = 0
            state = env.reset()
            done = False
            state_list = []
            action_list = []
            next_state_list = []
            done_list = []
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            gail.learn(expert_s, expert_a, state_list, action_list,
                       next_state_list, done_list)
            if (i + 1) % 10 == 0:
                pbar.set_postfix(
                    {'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(return_list)))
    plt.plot(iteration_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GAIL on {}'.format(env_name))
    plt.show()
