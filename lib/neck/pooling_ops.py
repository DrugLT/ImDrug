import torch
import torch.nn as nn
import torch.nn.functional as F


class GAP(nn.Module):
    """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """

    def __init__(self):
        super(GAP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs, input_kws):
        x = self.avgpool(inputs)
        #         x = x.view(x.shape[0], -1)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs, input_kws):
        return inputs

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
    
    def forward(self, inputs:list, input_kws:list):
        return torch.cat(inputs, 1)