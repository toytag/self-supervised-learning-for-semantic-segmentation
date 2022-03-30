import torch
import torch.nn as nn

from ..utils import ConvModule


class _PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        in_channel (int): Input number of channels.
        out_channel (int): Output number of channels.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid Module.
    """

    def __init__(self, in_channel, out_channel, pool_scales):
        super().__init__()
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                ConvModule(in_channel, out_channel, 1, bias=False),))

    def forward(self, x):
        """Forward function."""
        ppm_outs = [
            nn.functional.interpolate(
                ppm(x), size=x.size()[2:], mode='bilinear')
            for ppm in self
        ]
        return ppm_outs


class PSPHead(nn.Module):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`.

    Args:
        in_channel (int): Input number of channels. Default: 2048.
        out_channel (int): Output number of channels. Default: 512.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid Module.
            Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channel=2048, out_channel=512, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        assert isinstance(in_channel, int)
        assert isinstance(out_channel, int)
        assert isinstance(pool_scales, (list, tuple))
        self.out_channel = out_channel
        self.pool_scales = pool_scales
        self.psp_modules = _PPM(in_channel, out_channel, pool_scales)
        self.bottleneck = ConvModule(
            in_channel + len(pool_scales) * out_channel, out_channel, 3, padding=1, bias=False)

    def forward(self, inputs):
        """Forward function."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output
