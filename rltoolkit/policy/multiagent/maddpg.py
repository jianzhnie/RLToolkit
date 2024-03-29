from copy import deepcopy

import torch
import torch.nn.functional as F

from rltoolkit.models.utils import check_model_method
from rltoolkit.policy.base_policy import Algorithm

__all__ = ['MADDPG']


class MADDPG(Algorithm):

    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """MADDPG algorithm.

        Args:
            model (Model): forward network of actor and critic.
                                The function get_actor_params() of model should be implemented.
            agent_index (int): index of agent, in multiagent env
            act_space (list): action_space, gym space
            gamma (float): discounted factor for reward computation.
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            critic_lr (float): learning rate of the critic model
            actor_lr (float): learning rate of the actor model
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(agent_index, int)
        assert isinstance(act_space, list)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.continuous_actions = False
        if not len(act_space) == 0 and hasattr(act_space[0], 'high') \
                and not hasattr(act_space[0], 'num_discrete_space'):
            self.continuous_actions = True

        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.target_model = deepcopy(model)
        self.sync_target(0)

        self.actor_optimizer = torch.optim.Adam(
            lr=self.actor_lr, params=self.model.get_actor_params())
        self.critic_optimizer = torch.optim.Adam(
            lr=self.critic_lr, params=self.model.get_critic_params())

    def predict(self, obs):
        """use the policy model to predict actions.

        Args:
            obs (torch tensor): observation, shape([B] + shape of obs_n[agent_index])

        Returns:
            act (torch tensor): action, shape([B] + shape of act_n[agent_index]),
                noted that in the discrete case we take the argmax along the last axis as action
        """
        policy = self.model.policy(obs)

        if self.continuous_actions:
            action = policy[0]
            action = torch.tanh(action)
        else:
            action = F.softmax(policy, dim=-1)

        return action

    def sample(self, obs, use_target_model=False):
        """use the policy model to sample actions.

        Args:
            obs (torch tensor): observation, shape([B] + shape of obs_n[agent_index])
            use_target_model (bool): use target_model or not

        Returns:
            act (torch tensor): action, shape([B] + shape of act_n[agent_index]),
                noted that in the discrete case we take the argmax along the last axis as action
        """
        if use_target_model:
            policy = self.target_model.policy(obs)
        else:
            policy = self.model.policy(obs)

        # add noise for action exploration
        if self.continuous_actions:
            random_normal = torch.randn(size=policy[0].shape).to(self.device)
            action = policy[0] + torch.exp(policy[1]) * random_normal
            action = torch.tanh(action)
        else:
            uniform = torch.rand_like(policy)
            soft_uniform = torch.log(-1.0 * torch.log(uniform)).to(self.device)
            action = F.softmax(policy - soft_uniform, dim=-1)

        return action

    def Q(self, obs_n, act_n, use_target_model=False):
        """use the value model to predict Q values.

        Args:
            obs_n (list of torch tensor): all agents' observation, len(agent's num) + shape([B] + shape of obs_n)
            act_n (list of torch tensor): all agents' action, len(agent's num) + shape([B] + shape of act_n)
            use_target_model (bool): use target_model or not

        Returns:
            Q (torch tensor): Q value of this agent, shape([B])
        """
        if use_target_model:
            return self.target_model.value(obs_n, act_n)
        else:
            return self.model.value(obs_n, act_n)

    def learn(self, obs_n, act_n, target_q):
        """update actor and critic model with MADDPG algorithm."""
        critic_cost = self._critic_learn(obs_n, act_n, target_q)
        self.sync_target()
        return critic_cost

    def _actor_learn(self, obs_n, act_n):
        i = self.agent_index

        sample_this_action = self.sample(obs_n[i])
        action_input_n = act_n + []
        action_input_n[i] = sample_this_action
        eval_q = self.Q(obs_n, action_input_n)
        act_cost = torch.mean(-1.0 * eval_q)

        this_policy = self.model.policy(obs_n[i])
        # when continuous, 'this_policy' will be a tuple with two element: (mean, std)
        if self.continuous_actions:
            this_policy = torch.cat(this_policy, dim=-1)
        act_reg = torch.mean(torch.square(this_policy))

        cost = act_cost + act_reg * 1e-3

        self.actor_optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.model.get_actor_params(), 0.5)
        self.actor_optimizer.step()
        return cost

    def _critic_learn(self, obs_n, act_n, target_q):
        pred_q = self.Q(obs_n, act_n)
        cost = F.mse_loss(pred_q, target_q)

        self.critic_optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.model.get_critic_params(), 0.5)
        self.critic_optimizer.step()
        return cost

    def sync_target(self, decay=None):
        """update the target network with the training network.

        Args:
            decay(float): the decaying factor while updating the target network with the training network.
                        0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
