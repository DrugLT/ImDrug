import torch
import os
import dgl
import random
import numpy as np
import torch.nn as nn

"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    if _use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)
    return torch.device("cuda:0" if _use_gpu else "cpu")

def global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # gpu
    torch.cuda.manual_seed_all(seed) # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def gpu_enabled():
    return _use_gpu

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()


def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    if isinstance(np_batch, dict):
        return {
            k: elem_or_tuple_to_variable(x)
            for k, x in filter_batch(np_batch)
            if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
        }
    else:
        return from_numpy(np_batch)

def init_weight(m, initrange=0.1, zero_bias=False):
    if hasattr(m, 'weight'):
        m.weight.data.uniform_(-initrange, initrange)
        if hasattr(m, 'bias') and zero_bias:
            m.bias.data.zero_()

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def move_to_device(obj, device=None):
    if (device is None):
        device = torch.device('cuda')
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        return obj.to(device)

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()


def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)


def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)

def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.
    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
