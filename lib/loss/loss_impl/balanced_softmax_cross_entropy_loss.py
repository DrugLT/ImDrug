'''
Author: lianglzeng lianglzeng@tencent.com
Date: 2022-05-21 23:57:19
LastEditors: lianglzeng lianglzeng@tencent.com
LastEditTime: 2022-05-21 23:57:19
FilePath: /DrugLT/lib/loss/loss_impl/balanced_softmax_cross_entropy_loss.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch.nn import functional as F

from ..loss_base import CrossEntropy

class BalancedSoftmaxCE(CrossEntropy):
    r"""
    References:
    Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.

    Equation: Loss(x, c) = -log(\frac{n_c*exp(x)}{sum_i(n_i*exp(i)})
    """

    def __init__(self, para_dict=None):
        super(BalancedSoftmaxCE, self).__init__(para_dict)
        self.bsce_weight = torch.FloatTensor(self.num_class_list).to(self.device)

    def forward(self, inputs, targets,  **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        logits = inputs + self.weight_list.unsqueeze(0).expand(inputs.shape[0], -1).log()
        loss = F.cross_entropy(input=logits, target=targets.long())
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int
        """
        if not self.drw:
            self.weight_list = self.bsce_weight
        else:
            self.weight_list = torch.ones(self.bsce_weight.shape).to(self.device)
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.bsce_weight
