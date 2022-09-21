import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (parameters_to_vector,
                                               vector_to_parameters)
from torch.optim import Adam


class PolicyNet(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class ValueNet(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ActorCritic(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int,
                 action_dim: int) -> None:
        super(ActorCritic).__init__()
        """Initialize."""

        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.value_net = ValueNet(state_dim, hidden_dim)

    def policy(self, x: torch.Tensor) -> torch.Tensor:
        out = self.policy_net(x)
        return out

    def value(self, x: torch.Tensor) -> torch.Tensor:
        out = self.value_net(x)
        return out


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class Agent(object):
    """A2CAgent interacting with environment. The “Critic” estimates the value
    function. This could be the action-value (the Q value) or state-value (the
    V value).

    The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients).

    Atribute:
        gamma (float): discount factor
        lmbda: (float): GAE参数
        gamma: (float), 折扣回报系数
        kl_constraint: (float) = 0.01,  KL距离最大限制
        backtrack_coeff: (float) = 0.5, 线性搜索参数
        backtrack_iter: (int) = 15,     线性搜索次数
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        device (torch.device): cpu / gpu
    """

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 critic_lr: float,
                 lmbda: float,
                 gamma: float,
                 algo: str = 'trpo',
                 train_critic_iters: int = 10,
                 kl_constraint: float = 0.01,
                 backtrack_coeff: float = 0.5,
                 backtrack_alpha: float = 0.5,
                 backtrack_iter: int = 15,
                 backtrack_iters: List[int] = [],
                 device: Any = None):

        self.lmbda = lmbda  # GAE参数
        self.gamma = gamma  # 折扣因子
        self.algo = algo  # 算法名
        self.train_critic_iters = train_critic_iters
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.backtrack_coeff = backtrack_coeff  # 线性搜索参数
        self.backtrack_alpha = backtrack_alpha
        self.backtrack_iter = backtrack_iter  # 线性搜索次数
        self.backtrack_iters = backtrack_iters  # 线性搜索次数

        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 价值网络优化器
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device

    def sample(self, obs: np.ndarray) -> Tuple[int, float]:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        prob = self.actor(obs)
        action_dist = Categorical(prob)
        action = action_dist.sample()
        return action.item()

    def predict(self, obs) -> int:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        # 根据动作概率选择概率最高的动作
        select_action = self.actor(obs).argmax().item()
        return select_action

    # have checked
    def hessian_vector_product(self,
                               obs: torch.Tensor,
                               old_action_dists: torch.Tensor,
                               vector: torch.Tensor,
                               damping_coeff: float = 0.1):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = Categorical(self.actor(obs))
        # 计算平均KL距离
        kl = torch.mean(kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(
            kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = self.flat_grad(kl_grad)
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        kl_hessian_gard = torch.autograd.grad(kl_grad_vector_product,
                                              self.actor.parameters())
        kl_hessian_vector = self.flat_grad(kl_hessian_gard, hessian=True)
        # kl_hessian_vector = torch.cat(
        #     [grad.view(-1) for grad in kl_hessian_gard])
        kl_hessian_vector = kl_hessian_vector + vector * damping_coeff
        return kl_hessian_vector

    # have checked
    def conjugate_gradient(self,
                           grad: torch.Tensor,
                           obs: torch.Tensor,
                           old_action_dists: torch.Tensor,
                           cg_iters: int = 10,
                           eps: float = 1e-8,
                           residual_tol: float = 1e-10):
        """Conjugate gradient algorithm  共轭梯度法求解方程 (see
        https://en.wikipedia.org/wiki/Conjugate_gradient_method)"""
        # from openai baseline code
        # https://github.com/openai/baselines/blob/master/baselines/common/cg.py

        x = torch.zeros_like(grad).to(self.device)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r).to(self.device)
        for _ in range(cg_iters):  # 共轭梯度主循环
            Hp = self.hessian_vector_product(obs, old_action_dists, p)
            alpha = rdotr / (torch.dot(p, Hp) + eps)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def flat_grad(self, grads, hessian=False):
        grad_flatten = []
        if hessian:
            for grad in grads:
                grad_flatten.append(grad.contiguous().view(-1))
            grad_flatten = torch.cat(grad_flatten)
        else:
            for grad in grads:
                grad_flatten.append(grad.view(-1))
            grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def compute_policy_obj(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantage: torch.Tensor,
        old_log_probs: torch.Tensor,
        actor: nn.Module,
    ):
        # 计算策略目标
        log_probs = torch.log(actor(obs).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantage: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_action_dists: torch.Tensor,
        grads_vector: torch.Tensor,
        descent_direction: torch.Tensor,
        step_size: torch.Tensor,
    ):  # 线性搜索
        old_params = parameters_to_vector(self.actor.parameters())
        old_policy_obj = self.compute_policy_obj(obs, actions, advantage,
                                                 old_log_probs, self.actor)
        expected_improve = (grads_vector * step_size * descent_direction).sum(
            0, keepdim=True)
        for i in range(1, self.backtrack_iter + 1):  # 线性搜索主循环
            # Backtracking line search
            # (https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) 464p.
            coef = self.backtrack_coeff**i
            new_params = old_params + coef * step_size * descent_direction
            new_actor = copy.deepcopy(self.actor)
            # 更新策略网络的参数
            vector_to_parameters(new_params, new_actor.parameters())
            new_action_dists = Categorical(new_actor(obs))
            kl_div = torch.mean(
                kl_divergence(old_action_dists, new_action_dists))
            # 重新计算策略目标
            new_policy_obj = self.compute_policy_obj(obs, actions, advantage,
                                                     old_log_probs, new_actor)

            loss_improve = new_policy_obj - old_policy_obj
            expected_improve *= self.backtrack_coeff
            improve_condition = loss_improve / expected_improve

            if improve_condition > self.backtrack_alpha and kl_div < self.kl_constraint:
                self.backtrack_iters.append(i)
                return new_params
        return old_params

    def policy_learn(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_action_dists: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantage: torch.Tensor,
        eps: float = 1e-8,
    ) -> None:
        old_policy_obj = self.compute_policy_obj(obs, actions, advantage,
                                                 old_log_probs, self.actor)
        # Symbols needed for Conjugate gradient solver
        grads = torch.autograd.grad(old_policy_obj, self.actor.parameters())
        # grad_vector = torch.cat([grad.view(-1) for grad in grads]).detach()
        grads_vector = self.flat_grad(grads)

        # 用共轭梯度法计算x = H^(-1)g
        # Core calculations for NPG or TRPO
        descent_direction = self.conjugate_gradient(grads_vector, obs,
                                                    old_action_dists)

        gHg = self.hessian_vector_product(obs, old_action_dists,
                                          descent_direction)
        gHg = torch.dot(gHg, descent_direction).sum(0)
        # 对梯度下降的半径进行限制
        step_size = torch.sqrt(2 * self.kl_constraint / (gHg + eps))
        if self.algo == 'npg':
            pass

        elif self.algo == 'trpo':
            # 线性搜索得到新的参数
            # Backtracking line search
            new_params = self.line_search(obs, actions, advantage,
                                          old_log_probs, old_action_dists,
                                          grads_vector, descent_direction,
                                          step_size)
            # 用线性搜索后的参数更新策略网络
            vector_to_parameters(new_params, self.actor.parameters())

    def learn(self, transition_dict: Dict[str, list]) -> None:
        """Update the model by gradient descent."""
        obs = torch.tensor(
            transition_dict['obs'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype=torch.float).view(-1, 1).to(self.device)
        next_obs = torch.tensor(
            transition_dict['next_obs'], dtype=torch.float).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'],
            dtype=torch.float).view(-1, 1).to(self.device)

        # Update value network parameter
        # 更新价值网络参数
        for _ in range(self.train_critic_iters):
            td_target = rewards + self.gamma * self.critic(next_obs) * (1 -
                                                                        dones)
            td_delta = td_target - self.critic(obs)

            critic_loss = F.mse_loss(self.critic(obs), td_target.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()  # 更新价值函数

        # compute anvantage
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        # Prediction logπ_old(s), logπ(s)
        old_log_probs = torch.log(self.actor(obs).gather(1, actions)).detach()
        old_action_dists = Categorical(self.actor(obs).detach())

        # 更新策略函数
        self.policy_learn(obs, actions, old_action_dists, old_log_probs,
                          advantage)

        return critic_loss, critic_loss
