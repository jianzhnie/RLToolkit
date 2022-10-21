import copy

import torch
import torch.optim as optim

from rltoolkit.models.noisynet import NoisyDulingNet, NoisyNet
from rltoolkit.utils.scheduler import LinearDecayScheduler
from rltoolkit.utils.utils import check_model_method


class Agent(object):
    """Agent.

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
        action_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        start_lr (float): initial learning rate
        update_target_step (int): target network update frequency
    """

    def __init__(
            self,
            model_name: str,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            total_step: int,
            update_target_step: int,
            start_lr: float = 0.001,
            end_lr: float = 0.00001,
            gamma: float = 0.99,
            batch_size: int = 64,
            # Categorical DQN parameters
            std_init: float = 0.1,
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 1,
            device: str = 'cpu'):
        super().__init__()
        self.model_name = model_name
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.end_lr = end_lr
        self.gamma = gamma
        self.batch_size = batch_size
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        # N-step Learning
        self.n_step = n_step
        self.device = device
        self.support = torch.linspace(self.v_min, self.v_max,
                                      self.atom_size).to(self.device)

        if model_name == 'noisynetwork':
            print('Using NoisyNet')
            self.model = NoisyNet(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                std_init=std_init)
        elif model_name == 'noisydulingnetwork':
            print('Using NoisyDulingNetwork')
            self.model = NoisyDulingNet(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                atom_size=atom_size,
                std_init=std_init,
                support=self.support)
        self.target_model = copy.deepcopy(self.model)
        check_model_method(self.model, 'forward', self.__class__.__name__)
        self.smoothl1_loss = torch.nn.SmoothL1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=start_lr)
        self.lr_scheduler = LinearDecayScheduler(start_lr, total_step)

    def sample(self, obs) -> int:
        """Sample an action when given an observation,
        NoisyNet: no epsilon greedy action selection

        Args:
            obs: current observation

        Returns:
            act (int): action
        """
        act = self.predict(obs)
        return act

    def predict(self, obs) -> int:
        """Predict an action when given an observation, a greedy action will be
        returned.

        Args:
            obs (np.float32): shape of (batch_size, obs_dim) , current observation

        Returns:
            act(int): action
        """
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        selected_action = self.model(obs).argmax().item()
        return selected_action

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
        action = torch.tensor(action, dtype=torch.long).to(self.device)

        if self.global_update_step % self.update_target_step == 0:
            self.model.sync_weights_to(self.target_model)

        # calculate dqn loss
        # 1-step Learning loss
        if self.model_name == 'noisydulingnetwork':
            loss = self._compute_dis_dqn_loss(obs, action, reward, next_obs,
                                              terminal)

        elif self.model_name == 'noisynetwork':
            loss = self._cumpute_noisy_dqn_loss(obs, action, reward, next_obs,
                                                terminal)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_update_step += 1
        # NoisyNet: reset noise
        self.model.reset_noise()
        self.target_model.reset_noise()
        # learning rate decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.lr_scheduler.step(1), self.end_lr)
        return loss.item()

    def _cumpute_noisy_dqn_loss(self, obs: torch.Tensor, action: torch.Tensor,
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
        # calculate dqn loss
        pred_value = self.model(obs).gather(1, action)
        # model for selection actions.
        greedy_action = self.model(next_obs).max(dim=1, keepdim=True)[1]
        with torch.no_grad():
            # target_model for evaluation.
            next_q_value = self.target_model(next_obs).gather(1, greedy_action)
            if self.n_step > 1:
                gamma = self.gamma**self.n_step
            else:
                gamma = self.gamma
            target = reward + (1 - terminal) * gamma * next_q_value
        loss = self.mse_loss(pred_value, target)
        return loss

    def _compute_dis_dqn_loss(self, obs: torch.Tensor, action: torch.Tensor,
                              reward: torch.Tensor, next_obs: torch.Tensor,
                              terminal: torch.Tensor) -> torch.Tensor:
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.model(next_obs).argmax(1)
            next_dist = self.target_model.dist(next_obs)
            next_dist = next_dist[range(self.batch_size), next_action]

            if self.n_step > 1:
                gamma = self.gamma**self.n_step
            else:
                gamma = self.gamma
            t_z = reward + (1 - terminal) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.atom_size,
                               self.batch_size).long().unsqueeze(1).expand(
                                   self.batch_size,
                                   self.atom_size).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                          (next_dist *
                                           (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                          (next_dist *
                                           (b - l.float())).view(-1))

        dist = self.model.dist(obs)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)
        loss = torch.mean(elementwise_loss)
        return loss
