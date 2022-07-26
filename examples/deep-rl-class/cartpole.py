from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):

    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma,
              print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        # Here, we calculate discounts for instance [0.99^1, 0.99^2, 0.99^3, ..., 0.99^len(rewards)]
        discounts = [gamma**i for i in range(len(rewards) + 1)]
        # We calculate the return by sum(gamma[t] * reward[t])
        R = sum([a * b for a, b in zip(discounts, rewards)])

        # Line 7:
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))

    return scores


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """Evaluate the agent for ``n_eval_episodes`` episodes and returns average
    reward and std of reward.

    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env_id = 'CartPole-v1'
    # Create the env
    env = gym.make(env_id)

    # Create the evaluation env
    eval_env = gym.make(env_id)

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print('_____OBSERVATION SPACE_____ \n')
    print('The State Space is: ', s_size)
    print('Sample observation',
          env.observation_space.sample())  # Get a random observation

    print('\n _____ACTION SPACE_____ \n')
    print('The Action Space is: ', a_size)
    print('Action Space Sample',
          env.action_space.sample())  # Take a random action

    cartpole_hyperparameters = {
        'h_size': 16,
        'n_training_episodes': 1000,
        'n_evaluation_episodes': 10,
        'max_t': 1000,
        'gamma': 1.0,
        'lr': 1e-2,
        'env_id': env_id,
        'state_space': s_size,
        'action_space': a_size,
    }

    # Create policy and place it to the device
    cartpole_policy = Policy(cartpole_hyperparameters['state_space'],
                             cartpole_hyperparameters['action_space'],
                             cartpole_hyperparameters['h_size']).to(device)
    cartpole_optimizer = optim.Adam(
        cartpole_policy.parameters(), lr=cartpole_hyperparameters['lr'])

    scores = reinforce(cartpole_policy, cartpole_optimizer,
                       cartpole_hyperparameters['n_training_episodes'],
                       cartpole_hyperparameters['max_t'],
                       cartpole_hyperparameters['gamma'], 100)

    evaluate_agent(eval_env, cartpole_hyperparameters['max_t'],
                   cartpole_hyperparameters['n_evaluation_episodes'],
                   cartpole_policy)
