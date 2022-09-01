'''
Author: jianzhnie@126.com
Date: 2022-09-01 11:10:45
LastEditors: jianzhnie@126.com
LastEditTime: 2022-09-01 14:41:21
Description:

Copyright (c) 2022 by jianzhnie jianzhnie@126.com, All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

modified from https://github.com/marlbenchmark/on-policy

'''
import warnings

warnings.simplefilter('default')

import os

import torch
from algorithm import Algorithm

__all__ = ['Agent']
torch.set_num_threads(1)


class AgentBase(object):
    """`AgentBase` is the base class of the `parl.Agent` in different
    frameworks.

    `parl.Agent` is responsible for the general data flow outside the algorithm.
    """

    def __init__(self, algorithm):
        """

        Args:
            algorithm (`AlgorithmBase`): an instance of `AlgorithmBase`
        """
        self.alg = algorithm

    def get_weights(self, *args, **kwargs):
        """Get weights of the agent.

        Returns:
            (Dict): Dict of weights ({attribute name: numpy array/List/Dict})
        """
        return self.alg.get_weights(*args, **kwargs)

    def set_weights(self, weights, *args, **kwargs):
        """Set weights of the agent with given weights.

        Args:
            weights (Dict): Dict of weights
        """
        self.alg.set_weights(weights, *args, **kwargs)

    def learn(self, *args, **kwargs):
        """The training interface for Agent.

        This function will usually do the following things:
            1. Accept numpy data as input;
            2. Feed numpy data or onvert numpy data to tensor (optional);
            3. Call learn function in `Algorithm`.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict the action when given the observation of the enviroment.

        In general, this function is used in test process.

        This function will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data or onvert numpy data to tensor (optional);
           3. Call predict function in `Algorithm`.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Sample the action when given the observation of the enviroment.

        In general, this function is used in train process.

        This function will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data or onvert numpy data to tensor (optional);
           3. Call predict or sample function in `Algorithm`;
           4. Add sampling operation in numpy level. (unnecessary if sampling operation have done in `Algorithm`).
        """
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """Set the model in training mode."""
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        """Set the model in evaluation mode."""
        raise NotImplementedError


class Agent(AgentBase):
    """| `alias`: ``parl.Agent``

    | `alias`: ``parl.core.torch.agent.Agent``

    | Agent is one of the three basic classes of PARL.

    | It is responsible for interacting with the environment and collecting data for training the policy.
    | To implement a customized ``Agent``, users can:

      .. code-block:: python

        import parl

        class MyAgent(parl.Agent):
            def __init__(self, algorithm, act_dim):
                super(MyAgent, self).__init__(algorithm)
                self.act_dim = act_dim

    Attributes:
        device (torch.device): select GPU/CPU to be used.
        alg (parl.Algorithm): algorithm of this agent.

    Public Functions:
        - ``sample``: return a noisy action to perform exploration according to the policy.
        - ``predict``: return an estimate Q function given current observation.
        - ``learn``: update the parameters of self.alg.
        - ``save``: save parameters of the ``agent`` to a given path.
        - ``restore``: restore previous saved parameters from a given path.
        - ``train``: set the agent in training mode.
        - ``eval``: set the agent in evaluation mode.

    Todo:
        - allow users to get parameters of a specified model by specifying the model's name in ``get_weights()``.
    """

    def __init__(self, algorithm):
        """.

        Args:
            algorithm (parl.Algorithm): an instance of `parl.Algorithm`. This algorithm is then passed to `self.alg`.
            device (torch.device): specify which GPU/CPU to be used.
        """

        assert isinstance(algorithm, Algorithm)
        super(Agent, self).__init__(algorithm)
        # agent mode (bool): True is in training mode, False is in evaluation mode.
        self.training = True

    def learn(self, *args, **kwargs):
        """The training interface for ``Agent``.

        It is often used in the training stage.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict an estimated Q value when given the observation of the
        environment.

        It is often used in the evaluation stage.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Return an action with noise when given the observation of the
        environment.

        In general, this function is used in train process as noise is added to the action to preform exploration.
        """
        raise NotImplementedError

    def save(self, save_path, model=None):
        """Save parameters.

        Args:
            save_path(str): where to save the parameters.
            model(parl.Model): model that describes the neural network structure. If None, will use self.alg.model.

        Raises:
            ValueError: if model is None and self.alg.model does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model.ckpt')
        """
        if model is None:
            model = self.alg.model
        sep = os.sep
        dirname = sep.join(save_path.split(sep)[:-1])
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(model.state_dict(), save_path)

    def restore(self, save_path, model=None, map_location=None):
        """Restore previously saved parameters. This method requires a model
        that describes the network structure. The save_path argument is
        typically a value previously passed to ``save()``.

        Args:
            save_path(str): path where parameters were previously saved.
            model(parl.Model): model that describes the neural network structure. If None, will use self.alg.model.
            map_location: a function, torch.device, string or a dict specifying how to remap storage locations

        Raises:
            ValueError: if model is None and self.alg does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model.ckpt')
            agent.restore('./model.ckpt')

            agent.restore('./model.ckpt', map_location=torch.device('cpu')) # load gpu-trained model in cpu machine
        """

        if model is None:
            model = self.alg.model
        checkpoint = torch.load(save_path, map_location=map_location)
        model.load_state_dict(checkpoint)

    def train(self):
        """Sets the agent in training mode, which is the default setting. Model
        of agent will be affected if it has some modules (e.g. Dropout,
        BatchNorm) that behave differently in train/evaluation mode.

        Example:

        .. code-block:: python

            agent.train()   # default setting
            assert (agent.training is True)
            agent.eval()
            assert (agent.training is False)
        """
        self.alg.model.train()
        self.training = True

    def eval(self):
        """Sets the agent in evaluation mode."""
        self.alg.model.eval()
        self.training = False
