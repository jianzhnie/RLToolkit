'''
Author: jianzhnie@126.com
Date: 2022-09-01 10:59:20
LastEditors: jianzhnie@126.com
LastEditTime: 2022-09-01 11:17:46
Description:

Copyright (c) 2022 by jianzhnie jianzhnie@126.com, All Rights Reserved.
'''
import torch
import torch.nn as nn

__all__ = ['Model']


class ModelBase(object):
    """`ModelBase` is the base class of the `rltoolkit.Model` in different
    frameworks.

    This base class mainly do the following things:
        1. Implements APIs to manage model_id of the `rltoolkit.Model`;
        2. Defines common APIs that `rltoolkit.Model` should implement in different frameworks.
    """

    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        """Define forward network of the model."""
        raise NotImplementedError

    def get_weights(self, *args, **kwargs):
        """Get weights of the model."""
        raise NotImplementedError

    def set_weights(self, weights, *args, **kwargs):
        """Set weights of the model with given weights."""
        raise NotImplementedError

    def sync_weights_to(self, other_model):
        """Synchronize weights of the model to another model."""
        raise NotImplementedError

    def parameters(self):
        """Get the parameters of the model."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Call forward function."""
        return self.forward(*args, **kwargs)


class Model(nn.Module, ModelBase):
    """| `alias`: ``rltoolkit.Model``

    | `alias`: ``rltoolkit.core.torch.agent.Model``

    | ``Model`` is a base class of rltoolkit for the neural network. A ``Model`` is
    usually a policy or Q-value function, which predicts an action or an
    estimate according to the environmental observation.

    | To use the ``PyTorch`` backend model, user needs to call ``super(Model,
    self).__init__()`` at the beginning of ``__init__`` function.

    | ``Model`` supports duplicating a ``Model`` instance in a pythonic way:

    | ``copied_model = copy.deepcopy(model)``

    Example:

    .. code-block:: python

        import rltoolkit
        import torch.nn as nn

        class Policy(rltoolkit.Model):
            def __init__(self):
                super(Policy, self).__init__()
                self.fc = nn.Linear(in_features=100, out_features=32)

            def policy(self, obs):
                out = self.fc(obs)
                return out

        policy = Policy()
        copied_policy = copy.deepcopy(model)

    Attributes:
        model_id(str): each model instance has its unique model_id.

    Public Functions:
        - ``sync_weights_to``: synchronize parameters of the current model to
        another model.
        - ``get_weights``: return a dict containing all the parameters of the
        current model.
        - ``set_weights``: copy parameters from ``set_weights()`` to the model.
        - ``forward``: define the computations of a neural network. **Should**
        be overridden by all subclasses.
    """

    def __init___(self):
        super(Model, self).__init__()

    def sync_weights_to(self, target_model, decay=0.0):
        """Synchronize parameters of current model to another model.

        target_model_weights = decay * target_model_weights + (1 - decay) *
        current_model_weights

        Args:
            target_model (`rltoolkit.Model`): an instance of ``Model`` that has the
            same neural network architecture as the current model.
            decay (float):  the rate of decline in copying parameters. 0 if no
            parameters decay when synchronizing the parameters.

        Example:

        .. code-block:: python

            import copy
            # create a model that has the same neural network structures.
            target_model = copy.deepcopy(model)

            # after initializing the parameters ...
            model.sync_weights_to(target_model)

        Note:
            Before calling ``sync_weights_to``, parameters of the model must
            have been initialized.
        """

        assert target_model is not self, 'cannot copy between identical model'
        assert isinstance(target_model, Model)
        assert self.__class__.__name__ == target_model.__class__.__name__, \
            'must be the same class for params syncing!'
        assert (decay >= 0 and decay <= 1)

        target_vars = dict(target_model.named_parameters())
        for name, var in self.named_parameters():
            target_vars[name].data.copy_(decay * target_vars[name].data +
                                         (1 - decay) * var.data)

    def get_weights(self):
        """Returns a Python dict containing parameters of current model.

        Returns: a Python dict containing the parameters of current model.
        """
        weights = self.state_dict()
        for key in weights.keys():
            weights[key] = weights[key].cpu().numpy()
        return weights

    def set_weights(self, weights):
        """Copy parameters from ``set_weights()`` to the model.

        Args:
            weights (dict): a Python dict containing the parameters.
        """
        new_weights = dict()
        for key in weights.keys():
            new_weights[key] = torch.from_numpy(weights[key])
        self.load_state_dict(new_weights)
