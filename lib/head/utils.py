import numpy as np
import torch
from torch import nn

from layer import SiLU
from layer import TheLayerNorm

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function. #TODO: the type of activation should be updated.
    :return: The activation function module.
    """
    if type(activation) is not str:
        return activation
    activation = activation.lower()
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == "silu":
        return SiLU()
    elif activation is None:
        return nn.Identity()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')