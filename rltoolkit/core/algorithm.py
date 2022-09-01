'''
Author: jianzhnie@126.com
Date: 2022-09-01 10:58:47
LastEditors: jianzhnie@126.com
LastEditTime: 2022-09-01 12:30:20
Description:

Copyright (c) 2022 by jianzhnie jianzhnie@126.com, All Rights Reserved.
'''

import warnings

warnings.simplefilter('default')

from .model import Model, ModelBase

__all__ = ['Algorithm']


class AlgorithmBase(object):
    """`AlgorithmBase` is the base class of the `parl.Algorithm` in different
    frameworks.

    This base class mainly do the following things:
        1. Implements APIs to set or get weights of all `ModelBase` in self.__dict__;
        2. Defines common APIs that `parl.Algorithm` should implement in different frameworks.
    """

    def __init__(self):
        pass

    def get_weights(self):
        """Get weights of all `ModelBase` in self.__dict__.

        Note:
            `ModelBase` in list, tuple and dict will be included. But `ModelBase` in
            nested list, tuple and dict won't be included.

        Returns:
            Dict of weights ({attribute name: numpy array/List/Dict})
        """
        model_weights = {}
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if isinstance(value, ModelBase):
                model_weights[key] = value.get_weights()
            elif isinstance(value, list) or isinstance(value, tuple):
                weights_list = []
                for x in value:
                    if isinstance(x, ModelBase):
                        weights_list.append(x.get_weights())
                if weights_list:
                    model_weights[key] = weights_list
            elif isinstance(value, dict):
                weights_dict = {}
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, ModelBase):
                        weights_dict[sub_k] = sub_v.get_weights()
                if weights_dict:
                    model_weights[key] = weights_dict
        return model_weights

    def set_weights(self, weights):
        """Set weights of all `ModelBase` in self.__dict__.

        Note:
            `ModelBase` in list, tuple and dict will be included. But `ModelBase` in
            nested list, tuple and dict won't be included.

        Args:
            weights (Dict): Dict of weights ({attribute name: numpy array/List/Dict})
        """
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if isinstance(value, ModelBase):
                assert key in weights, 'weights is inconsistent with current algorithm.'
                value.set_weights(weights[key])
            elif isinstance(value, list) or isinstance(value, tuple):
                model_list = []
                for x in value:
                    if isinstance(x, ModelBase):
                        model_list.append(x)
                if model_list:
                    assert key in weights and len(model_list) == len(weights[key]), \
                        'weights is inconsistent with current algorithm.'
                    for i, model in enumerate(model_list):
                        model.set_weights(weights[key][i])
            elif isinstance(value, dict):
                model_dict = {}
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, ModelBase):
                        model_dict[sub_k] = sub_v
                if model_dict:
                    assert key in weights and set(model_dict.keys()) == set(weights[key].keys()), \
                        'weights is inconsistent with current algorithm.'
                    for sub_k, model in model_dict.items():
                        model.set_weights(weights[key][sub_k])

    def learn(self, *args, **kwargs):
        """define learning process, such as how to optimize the model."""
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """define predicting process, such as using policy model to predict
        actions when given observations."""
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """define sampling process, such as using policy model to sample
        actions when given observations."""
        raise NotImplementedError


class Algorithm(AlgorithmBase):
    """| `alias`: ``parl.Algorithm``

    | `alias`: ``parl.core.torch.algorithm.Algorithm``

    | ``Algorithm`` defines the way how to update the parameters of the
    ``Model``. This is where we define loss functions and the optimizer of the
    neural network. An ``Algorithm`` has at least a model.

    | PARL has implemented various algorithms(DQN/DDPG/PPO/A3C/IMPALA) that can
    be reused quickly, which can be accessed with ``parl.algorithms``.

    Example:

    .. code-block:: python

        import parl

        model = Model()
        dqn = parl.algorithms.DQN(model, lr=1e-3)

    Attributes:
        model(``parl.Model``): a neural network that represents a policy or a
        Q-value function.

    Pulic Functions:
        - ``predict``: return an estimate q value given current observation.
        - ``learn``: define the loss function and create an optimizer to
        minimize the loss.
    """

    def __init__(self, model=None):
        """
        Args:
            model(``parl.Model``): a neural network that represents a policy or
            a Q-value function.
        """
        assert isinstance(model, Model)
        self.model = model

    def get_weights(self):
        """Get weights of self.model.

        Returns:
            weights (dict): a Python dict containing the parameters of
            self.model.
        """
        return self.model.get_weights()

    def set_weights(self, params):
        """Set weights from ``get_weights`` to the model.

        Args:
            weights (dict): a Python dict containing the parameters of
            self.model.
        """
        self.model.set_weights(params)

    def learn(self, *args, **kwargs):
        """Define the loss function and create an optimizer to minimize the
        loss."""
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Refine the predicting process, e.g,.

        use the policy model to predict actions.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """define sampling process, such as using policy model to sample
        actions when given observations."""
        raise NotImplementedError
