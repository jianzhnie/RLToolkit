import random
import sys
from collections import deque

import numpy as np
import torch

from simplerl.models.ddqn import MarioNet

sys.path.append('../../')


class Mario(object):
    """Act according to the optimal action policy based on the current state
    (of the environment).

    Remember experiences. Experience = (current state, current action, reward, next state). Mario caches and later recalls \
        his experiences to update his action policy.
    Learn a better action policy over time
    """
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e5  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.save_every = 5e5  # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """Given a state, choose an epsilon-greedy action and update value of
        step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        state = state.__array__()
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(
                state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min,
                                    self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.FloatTensor(state).cuda()
            next_state = torch.FloatTensor(next_state).cuda()
            action = torch.LongTensor([action]).cuda()
            reward = torch.FloatTensor([reward]).cuda()
            done = torch.BoolTensor([done]).cuda()
        else:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            action = torch.LongTensor([action])
            reward = torch.FloatTensor([reward])
            done = torch.BoolTensor([done])

        self.memory.append((
            state,
            next_state,
            action,
            reward,
            done,
        ))

    def recall(self):
        """Retrieve a batch of experiences from memory."""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(
        ), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state,
                             model='online')[np.arange(0, self.batch_size),
                                             action]
        # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state,
                          model='target')[np.arange(0, self.batch_size),
                                          best_action]
        td_target = (reward + (1 - done.float()) * self.gamma * next_Q).float()
        return td_target

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self):
        save_path = self.save_dir / f'mario_net_{int(self.curr_step // self.save_every)}.chkpt'
        torch.save(
            dict(model=self.net.state_dict(),
                 exploration_rate=self.exploration_rate), save_path)
        print(f'MarioNet saved to {save_path} at step {self.curr_step}')

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f'{load_path} does not exist')

        ckp = torch.load(load_path,
                         map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(
            f'Loading model at {load_path} with exploration rate {exploration_rate}'
        )
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
