import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from ..utils import get_one_hot
from ..loss_base import CrossEntropy


class FocalR(nn.Module):
    def __init__(self, para_dict=None):
        super(FocalR, self).__init__()
        self.beta = para_dict['cfg']['loss']['FocalR']['BETA']
        self.gamma = para_dict['cfg']['loss']['FocalR']['GAMMA']
        self.choice = para_dict['cfg']['loss']['FocalR']['choice']

    def forward(self, inputs, targets, **kwargs):
        if len(inputs.shape) >= 2:
            inputs = inputs[..., 0]

        if self.choice == 'mse':
            return weighted_focal_mse_loss(inputs, targets)
        elif self.choice == 'l1':
            return weighted_focal_l1_loss(inputs, targets)
        elif self.choice == 'huber':
            return weighted_huber_loss(inputs, targets)
        else:
            raise ValueError("The hyperparameter 'choice' of FocalR must be one of ['mse', 'l1', 'huber'].")

    def update(self, epoch):
        pass



def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    # loss = F.l1_loss(inputs, targets, reduction='none')
    loss = torch.abs(inputs - targets)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (torch.sigmoid(beta * torch.abs(inputs - targets))) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss