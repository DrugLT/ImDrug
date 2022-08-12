import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from ..utils import get_one_hot
from ..loss_base import CrossEntropy


class BalancedMSELoss(nn.Module):
    def __init__(self, para_dict=None):
        super(BalancedMSELoss, self).__init__()
        self.sigma = para_dict['cfg']['loss']['BalancedMSELoss']['SIGMA']

    def forward(self, inputs, targets, **kwargs):
        if len(inputs.shape) >= 2:
            inputs = inputs[..., 0]

        inputs = inputs.reshape(-1, 1)
        targets = targets.reshape(-1, 1)
        logits = - 0.5 * (inputs - targets.T).pow(2) / (self.sigma ** 2)
        loss = F.cross_entropy(logits, torch.arange(inputs.shape[0]).cuda())
        loss = loss * (2 * (self.sigma ** 2))
        return loss

    def update(self, epoch):
        pass
