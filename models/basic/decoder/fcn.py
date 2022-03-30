import torch.nn as nn

from ..utils import ConvModule


class MoCoFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channel = 256
        self.convs = nn.Sequential(
            ConvModule(2048, 256, 3, padding=6, dilation=6, bias=False),
            ConvModule(256, 256, 3, padding=6, dilation=6, bias=False))

    def forward(self, inputs):
        """Forward function."""
        x = self.convs(inputs[-1])
        return x


class FCN32s(nn.Module):
    def __init__(self,  in_channels=(2048,), out_channel=256):
        super().__init__()
        assert len(in_channels) >= 1
        self.out_channel = out_channel
        self.conv32s = nn.Sequential(
            ConvModule(in_channels[-1], out_channel, 1, bias=False),
            ConvModule(out_channel, out_channel, 3, padding=1, bias=False),)

    def forward(self, inputs):
        """Forward function."""
        x = self.conv32s(inputs[-1])
        return x


class FCN16s(FCN32s):
    def __init__(self, in_channels=(1024, 2048), out_channel=256):
        super().__init__(in_channels, out_channel)
        assert len(in_channels) >= 2
        self.conv16s = nn.Sequential(
            ConvModule(in_channels[-2], out_channel, 1, bias=False),
            ConvModule(out_channel, out_channel, 3, padding=1, bias=False),)

    def forward(self, inputs):
        """Forward function."""
        x = self.conv32s(inputs[-1])
        x = self.conv16s(inputs[-2]) + \
            nn.functional.interpolate(
                x, size=inputs[-2].shape[2:], mode='bilinear')
        return x


class FCN8s(FCN16s):
    def __init__(self, in_channels=(512, 1024, 2048), out_channel=256):
        super().__init__(in_channels, out_channel)
        assert len(in_channels) >= 3
        self.conv8s = nn.Sequential(
            ConvModule(in_channels[-3], out_channel, 1, bias=False),
            ConvModule(out_channel, out_channel, 3, padding=1, bias=False),)

    def forward(self, inputs):
        """Forward function."""
        x = self.conv32s(inputs[-1])
        x = self.conv16s(inputs[-2]) + \
            nn.functional.interpolate(
                x, size=inputs[-2].shape[2:], mode='bilinear')
        x = self.conv8s(inputs[-3]) + \
            nn.functional.interpolate(
                x, size=inputs[-3].shape[2:], mode='bilinear')
        return x
