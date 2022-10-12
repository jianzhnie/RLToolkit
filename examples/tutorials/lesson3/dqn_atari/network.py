import torch
import torch.nn as nn


class AtariModel(nn.Module):
    """Neural Network to solve Atari problem.

    Args:
        act_dim (int): Dimension of action space.
        dueling (bool): True if use dueling architecture else False
    """

    def __init__(self, act_dim, dueling=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=1,
            padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.dueling = dueling

        if dueling:
            self.linear_1_adv = nn.Linear(in_features=6400, out_features=512)
            self.linear_2_adv = nn.Linear(
                in_features=512, out_features=act_dim)
            self.linear_1_val = nn.Linear(in_features=6400, out_features=512)
            self.linear_2_val = nn.Linear(in_features=512, out_features=1)

        else:
            self.linear_1 = nn.Linear(in_features=6400, out_features=act_dim)

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        """Perform forward pass.

        Args:
            obs (torch.Tensor): shape of (batch_size, 3, 84, 84), mini batch of observations
        """
        obs = obs / 255.0
        out = self.max_pool(self.relu(self.conv1(obs)))
        out = self.max_pool(self.relu(self.conv2(out)))
        out = self.max_pool(self.relu(self.conv3(out)))
        out = self.relu(self.conv4(out))
        out = self.flatten(out)

        if self.dueling:
            As = self.relu(self.linear_1_adv(out))
            As = self.linear_2_adv(As)
            V = self.relu(self.linear_1_val(out))
            V = self.linear_2_val(V)
            Q = As + (V - As.mean(dim=1, keepdim=True))

        else:
            Q = self.linear_1(out)

        return Q


class QNet(nn.Module):
    """Initialization.

    只有一层隐藏层的Q网络.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DulingNet(nn.Module):
    """只有一层隐藏层的A网络和V网络."""

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        """Initialization."""
        super(DulingNet, self).__init__()
        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        # set advantage layer
        self.advantage_layer = nn.Linear(hidden_dim, action_dim)
        # set value layer
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        # Q值由V值和A值计算得到
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q
