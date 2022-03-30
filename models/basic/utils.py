import torch.nn as nn


class ConvModule(nn.Sequential):
    """Conv + BN + ACT"""
    def __init__(self, in_channel, out_channel, kernel_size,
        conv=nn.Conv2d, bn=nn.BatchNorm2d, act=nn.ReLU, **kwargs):
        super().__init__()
        if conv is not None:
            self.add_module('conv', conv(in_channel, out_channel, kernel_size, **kwargs))
        if bn is not None:
            self.add_module('bn', bn(out_channel))
        if act is not None:
            self.add_module('act', act(inplace=True))

    def forward(self, x):
        """Forward function."""
        return super().forward(x)


class ConvMLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_size=2048):
        super().__init__(
            nn.Conv2d(input_dim, hidden_size, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, output_dim, 1))


class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_size=2048):
        super().__init__(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim))