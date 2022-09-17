import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import Model


# 定义一个添加噪声的网络层
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        """Initialization."""
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        # 向模块添加持久缓冲区,这通常用于注册不应被视为模型参数的缓冲区。
        # 例如，BatchNorm的running_mean不是一个参数，而是持久状态的一部分。
        # 缓冲区可以使用给定的名称作为属性访问。
        self.register_buffer('weight_epsilon',
                             torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode. It doesn't show remarkable difference of performance.
        """
        # if self.training:
        weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
        bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        # else:
        #     weight = self.weight_mu
        #     bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init /
                                     math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init /
                                   math.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class NoisyNet(Model):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        """Initialization."""
        super(NoisyNet, self).__init__()

        self.feature = nn.Linear(state_dim, hidden_dim)
        self.noisy_layer = NoisyLinear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        out = self.feature(x)
        out = self.relu(out)
        out = self.noisy_layer(out)
        return out

    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer.reset_noise()


class NoisyDulingNetwork(Model):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 atom_size: int, support: torch.Tensor):
        """Initialization."""
        super(NoisyDulingNetwork, self).__init__()

        self.support = support
        self.action_dim = action_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_layer = NoisyLinear(hidden_dim, action_dim * atom_size)

        # set value layer
        self.value_layer = NoisyLinear(hidden_dim, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        advantage = self.advantage_layer(feature).view(-1, self.action_dim,
                                                       self.atom_size)
        value = self.value_layer(feature).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_layer.reset_noise()
        self.value_layer.reset_noise()


class CategoryNetwork(Model):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 atom_size: int, support: torch.Tensor):
        """Initialization."""
        super(CategoryNetwork, self).__init__()

        self.support = support
        self.out_dim = action_dim
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * atom_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist
