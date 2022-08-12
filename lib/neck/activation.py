"""Element-wise SiLU non-linearity."""

from torch import nn


class SiLU(nn.Module):
    """Element-wise SiLU non-linearity."""

    @classmethod
    def forward(cls, x):
        """Perform the forward pass."""

        return x * nn.Sigmoid()(x)
 