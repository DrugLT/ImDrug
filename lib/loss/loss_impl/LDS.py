import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from ..loss_base import CrossEntropy


class LDS(nn.Module):
    def __init__(self, para_dict=None):
        super(LDS, self).__init__()
        self.base_loss = para_dict['cfg']['loss']['LDS']['base_loss']

    def forward(self, inputs, targets, lds_weight=None, **kwargs):
        if self.base_loss == 'mse':
            loss = F.mse_loss(inputs[..., 0].float(), targets.float(), reduction='none')
        elif self.base_loss == 'l1':
            loss = F.l1_loss(inputs[..., 0].float(), targets.float(), reduction='none')

        if lds_weight is not None:
            loss = loss * lds_weight.expand_as(loss)
        loss = torch.mean(loss)
        return loss

    def update(self, epoch):
        pass