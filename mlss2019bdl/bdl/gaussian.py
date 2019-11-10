import torch
import torch.nn.functional as F

from torch.nn import Linear, Conv2d

from .base import FreezableWeight, PenalizedWeight


class BaseGaussianLinear(Linear, FreezableWeight, PenalizedWeight):
    """Linear layer with Gaussian Mean Field weight distribution."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(
            torch.Tensor(*self.weight.shape))

        self.reset_variational_parameters()

    def reset_variational_parameters(self):
        self.log_sigma2.data.normal_(-5, 0.1)  # from arxiv:1811.00596

    def forward(self, input):
        """Forward pass for the linear layer with the local reparameterization trick."""

        if self.is_frozen():
            return F.linear(input, self.frozen_weight, self.bias)

        s2 = F.linear(input * input, torch.exp(self.log_sigma2), None)

        mu = super().forward(input)
        return mu + torch.randn_like(s2) * torch.sqrt(s2 + 1e-20)

    def freeze(self):

        with torch.no_grad():
            stdev = torch.exp(0.5 * self.log_sigma2)
            weight = torch.normal(self.weight, std=stdev)

        self.register_buffer("frozen_weight", weight)


class BaseGaussianConv2d(Conv2d, PenalizedWeight, FreezableWeight):
    """Convolutional layer with Gaussian Mean Field weight distribution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode)

        self.log_sigma2 = torch.nn.Parameter(
            torch.Tensor(*self.weight.shape))

        self.reset_variational_parameters()

    reset_variational_parameters = BaseGaussianLinear.reset_variational_parameters

    def forward(self, input):
        """Forward pass with the local reparameterization trick."""
        if self.is_frozen():
            return F.conv2d(input, self.frozen_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        s2 = F.conv2d(input * input, torch.exp(self.log_sigma2), None,
                      self.stride, self.padding, self.dilation, self.groups)

        mu = super().forward(input)
        return mu + torch.randn_like(s2) * torch.sqrt(s2 + 1e-20)

    freeze = BaseGaussianLinear.freeze


class GaussianLinearARD(BaseGaussianLinear):
    def penalty(self):
        # compute \tfrac12 \log (1 + \tfrac{\mu_{ji}}{\sigma_{ji}^2})
        log_weight2 = 2 * torch.log(torch.abs(self.weight) + 1e-20)

        # `softplus` is $x \mapsto \log(1 + e^x)$
        return 0.5 * torch.sum(F.softplus(log_weight2 - self.log_sigma2))


class GaussianConv2dARD(BaseGaussianConv2d):
    penalty = GaussianLinearARD.penalty
