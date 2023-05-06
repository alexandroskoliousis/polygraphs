"""
Tensor initializers
"""
import sys
import inspect
import torch

from .hyperparameters import HyperParameters


def uniform(size, params=None):
    """
    Fills tensor with values drawn from the uniform distribution U(a, b).
    """
    assert params and isinstance(params, HyperParameters)
    # Create empty tensor of given shape
    tensor = torch.empty(size)
    # Get relevant hyper-parameters
    lower = params.uniform.lower
    upper = params.uniform.upper
    # Fill tensor
    torch.nn.init.uniform_(tensor, a=lower, b=upper)
    return tensor


def gaussian(size, params=None):
    """
    Fills tensor with values drawn from a truncated guassian distribution.
    """
    assert params and isinstance(params, HyperParameters)
    # Create empty tensor of given shape
    tensor = torch.empty(size)
    # Fill auxiliary tensor (one column per attempt)
    aux = torch.empty(tensor.shape + (params.gaussian.attempts,))
    torch.nn.init.normal_(aux, mean=params.gaussian.mean, std=params.gaussian.std)
    # Find valid values
    valid = torch.gt(aux, params.gaussian.lower) & torch.lt(aux, params.gaussian.upper)
    # Pick valid values
    index = valid.max(-1, keepdim=True)[1]
    # Copy valid values to tensor
    tensor.data.copy_(aux.gather(-1, index).squeeze(-1))
    # Check if values have been successfully truncated
    # Long line subterfuge
    lower = params.gaussian.lower
    upper = params.gaussian.upper
    if not torch.all((tensor > lower) & (tensor < upper)):
        raise Exception(
            f"Failed to truncate N({params.gaussian.mean}, {params.gaussian.std}) "
            f"to ({lower}, {upper})"
        )
    return tensor


def constant(size, params=None):
    """
    Fills tensor with a value.
    """
    assert params and isinstance(params, HyperParameters)
    if isinstance(params.constant.value, list):
        assert size == (len(params.constant.value),)
        tensor = torch.Tensor(params.constant.value)
    else:
        # Create empty tensor of given shape
        tensor = torch.empty(size)
        torch.nn.init.constant_(tensor, params.constant.value)
    return tensor


def zeros(size, params=None):
    """
    Fills tensor with the value of 0.
    """
    # Unused parameter subterfuge
    del params
    # Create empty tensor of given shape
    tensor = torch.empty(size)
    torch.nn.init.zeros_(tensor)
    return tensor


def ones(size, params=None):
    """
    Fills tensor with the value of 1.
    """
    # Unused parameter subterfuge
    del params
    # Create empty tensor of given shape
    tensor = torch.empty(size)
    torch.nn.init.ones_(tensor)
    return tensor


def halfs(size, params=None):
    """
    Fills tensor with the value of 0.5.
    """
    # Unused parameter subterfuge
    del params
    # Create empty tensor of given shape
    tensor = torch.empty(size)
    torch.nn.init.constant_(tensor, 0.5)
    return tensor


def set_node_beliefs(beliefs, size, params):
    """
    Sets initial beliefs for specific nodes passed as a dictionary
    -- params.init.beliefs: { node: belief, ... }
    """
    if isinstance(params.beliefs, dict):
        # Update each node/key with value
        for k, v in params.beliefs.items():
            node = int(k)
            if node < size[0]:
                beliefs[node] = float(v)
            else:
                raise Exception(f"Error setting node belief: {k} out of bounds")

    return beliefs


def init(size, params):
    """
    Initializes values a tensor of given shape and parameters.
    """
    assert isinstance(params, HyperParameters)
    # Create friendly dictionary from list of (name, function) tuples
    members = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
    initializer = members.get(params.kind)
    if initializer is None:
        raise Exception(f"Invalid initializer type: {params.kind}")
    init_beliefs = initializer(size, params=params)
    # Set individual node beliefs
    beliefs = set_node_beliefs(init_beliefs, size, params)
    return beliefs
