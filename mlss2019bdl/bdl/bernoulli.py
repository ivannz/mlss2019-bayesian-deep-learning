import torch
import torch.nn.functional as F

from torch.nn import Linear

from .base import FreezableWeight, PenalizedWeight


class DropoutLinear(Linear, FreezableWeight):
    """Linear layer with dropout on inputs."""
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__(in_features, out_features, bias=bias)

        self.p = p

    def forward(self, input):
        if self.is_frozen():
            return F.linear(input, self.frozen_weight, self.bias)

        return super().forward(F.dropout(input, self.p, True))

    def freeze(self):
        # let's draw the new weight
        with torch.no_grad():
            prob = torch.full_like(self.weight[:1, :], 1 - self.p)
            feature_mask = torch.bernoulli(prob) / prob

            frozen_weight = self.weight * feature_mask

        # and store it
        self.register_buffer("frozen_weight", frozen_weight)


class DropoutConv2d(Conv2d, FreezableWeight):
    """2d Convolutional layer with dropout on input features."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 p=0.5):

        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode)

        self.p = p

    def forward(self, input):
        """Apply feature dropout and then forward pass through the convolution."""
        if self.is_frozen():
            return F.conv2d(input, self.frozen_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return super().forward(F.dropout2d(input, self.p, True))

    def freeze(self):
        """Sample the weight from the parameter distribution and freeze it."""
        prob = torch.full_like(self.weight[:1, :, :1, :1], 1 - self.p)
        feature_mask = torch.bernoulli(prob) / prob

        with torch.no_grad():
            frozen_weight = self.weight * feature_mask

        self.register_buffer("frozen_weight", frozen_weight)

