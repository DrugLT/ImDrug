import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from ..utils import get_one_hot
from ..loss_base import CrossEntropy


class FocalLoss(CrossEntropy):
    def __init__(self, para_dict=None):
        super(FocalLoss, self).__init__(para_dict)
        self.gamma = self.para_dict['cfg']['loss']['FocalLoss']['GAMMA'] #hyper-parameter

    def forward(self, inputs, targets, **kwargs):
        weight = (self.weight_list[targets]).to(targets.device) \
            if self.weight_list is not None else \
            torch.FloatTensor(torch.ones(targets.shape[0])).to(targets.device)
        preds = inputs.view(-1, inputs.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, targets.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1, targets.view(-1,1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = (loss * weight.view(-1, 1)).mean()
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = torch.FloatTensor(np.array([1 for _ in self.num_class_list])).to(self.device)
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)
            else:
                self.weight_list = torch.FloatTensor(np.array([1 for _ in self.num_class_list])).to(self.device)