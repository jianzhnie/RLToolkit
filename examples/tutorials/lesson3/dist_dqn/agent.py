import copy
import sys
import time

import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from network import DulingNet, QNet

sys.path.append('../../../../')
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from rltoolkit.data.buffer.replaybuffer import ReplayBuffer
from rltoolkit.models.utils import hard_target_update
from rltoolkit.utils.scheduler import LinearDecayScheduler


class Agent(object):
    """Agent.

    Args:
        action_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        learning_rate (float): initial learning rate
        update_target_step (int): target network update frequency
    """

    def __init__(self,
                 args,
                 envs: gym.Env,
                 algo: str = 'dqn',
                 total_steps: int = 100000,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 learning_rate: float = 0.001,
                 update_target_step: int = 100,
                 device='cpu'):
        super().__init__()

        self.args = args
        self.envs = envs
        self.algo = algo
        self.gamma = gamma
        self.epsilon = epsilon
        self.global_update_step = 0
        self.total_steps = total_steps
        self.update_target_step = update_target_step
        self.obs_dim = np.array(envs.single_observation_space.shape).prod()
        self.action_dim = envs.single_action_space.n
        self.replaybuffer = ReplayBuffer(
            buffer_size=args.memory_size,
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            n_envs=envs.num_envs,
            device=device,
            handle_timeout_termination=True,
        )

        # Main network
        if 'duling' in algo:
            self.qnet = DulingNet(self.obs_dim, args.hidden_dim,
                                  self.action_dim).to(device)
        else:
            self.qnet = QNet(self.obs_dim, args.hidden_dim,
                             self.action_dim).to(device)
        # Target network
        self.target_qnet = copy.deepcopy(self.qnet)
        # Create an optimizer
        self.optimizer = Adam(self.qnet.parameters(), lr=learning_rate)
        self.eps_scheduler = LinearDecayScheduler(epsilon, total_steps)
        self.lr_scheduler = LinearDecayScheduler(learning_rate, total_steps)

        run_name = args.run_name
        if self.args.use_wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f'runs/{run_name}')
        self.writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s' % ('\n'.join(
                [f'|{key}|{value}|' for key, value in vars(args).items()])),
        )

        self.device = device

    def select_action(self, obs) -> int:
        """Sample an action when given an observation, base on the current
        epsilon value, either a greedy action or a random action will be
        returned.

        Args:
            obs: current observation

        Returns:
            act (int): action
        """
        # Choose a random action with probability epsilon
        if np.random.rand() <= self.epsilon:
            act = np.array([
                self.envs.single_action_space.sample()
                for _ in range(self.envs.num_envs)
            ])
        else:
            # Choose the action with highest Q-value at the current state
            act = self.predict(obs)

        self.epsilon = max(self.eps_scheduler.step(1), self.args.min_epsilon)
        return act

    def predict(self, obs) -> int:
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim) , current observation

        Returns:
            act(int): action
        """
        if obs.ndim == 1:  # if obs is 1 dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        # action = self.qnet(obs).argmax().item()
        q_values = self.target_qnet(obs)
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions

    def learn(self, obs: torch.Tensor, action: torch.Tensor,
              reward: torch.Tensor, next_obs: torch.Tensor,
              terminal: torch.Tensor) -> float:
        """Update model with an episode data.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim)
            act (np.int32): shape of (batch_size)
            reward (np.float32): shape of (batch_size)
            next_obs (np.float32): shape of (batch_size, obs_dim)
            terminal (np.float32): shape of (batch_size)

        Returns:
            loss (float)
        """
        if self.global_update_step % self.update_target_step == 0:
            hard_target_update(self.qnet, self.target_qnet)

        # Prediction Q(s)
        pred_value = self.qnet(obs).gather(1, action)

        # Target for Q regression
        if self.algo in ['dqn', 'duling_dqn']:
            next_q_value = self.target_qnet(next_obs).max(1, keepdim=True)[0]

        elif self.algo in ['ddqn', 'duling_ddqn']:
            greedy_action = self.qnet(next_obs).max(dim=1, keepdim=True)[1]
            next_q_value = self.target_qnet(next_obs).gather(1, greedy_action)

        target = reward + (1 - terminal) * self.gamma * next_q_value

        # TD误差目标
        loss = F.mse_loss(pred_value, target)
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        loss.backward()
        # 反向传播更新参数
        self.optimizer.step()
        return loss.item()

    def train(self):
        start_time = time.time()
        obs = self.envs.reset()
        # Keep interacting until agent reaches a terminal state.
        while self.global_update_step < self.total_steps:
            self.global_update_step += 1
            # Collect experience (s, a, r, s') using some policy
            actions = self.select_action(obs)
            next_obs, rewards, dones, infos = self.envs.step(actions)
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if 'episode' in info.keys():
                    self.writer.add_scalar('charts/episodic_return',
                                           info['episode']['r'],
                                           self.global_update_step)
                    self.writer.add_scalar('charts/episodic_length',
                                           info['episode']['l'],
                                           self.global_update_step)
                    self.writer.add_scalar('charts/epsilon', self.epsilon,
                                           self.global_update_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]['terminal_observation']

            # Add experience to replay buffer
            self.replaybuffer.add(obs, real_next_obs, actions, rewards, dones,
                                  infos)
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # Start training when the number of experience is greater than batch_size
            if self.global_update_step > self.args.memory_warmup_size:
                samples = self.replaybuffer.sample(self.args.batch_size)
                batch_obs = samples.obs
                batch_action = samples.actions
                batch_reward = samples.rewards
                batch_next_obs = samples.next_obs
                batch_terminal = samples.dones

                loss = self.learn(batch_obs, batch_action, batch_reward,
                                  batch_next_obs, batch_terminal)

                if self.global_update_step % 100 == 0:
                    self.writer.add_scalar('losses/td_loss', loss,
                                           self.global_update_step)
                    print(
                        'SPS:',
                        int(self.global_update_step /
                            (time.time() - start_time)))
                    self.writer.add_scalar(
                        'charts/SPS',
                        int(self.global_update_step /
                            (time.time() - start_time)),
                        self.global_update_step)

        return 0
