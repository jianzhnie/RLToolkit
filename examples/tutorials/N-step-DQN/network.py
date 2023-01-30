import torch
import torch.nn as nn


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
