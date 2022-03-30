import torch
import torch.nn as nn

from ..utils import ConvModule


class _ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        in_channel (int): Input number of channels.
        out_channel (int): Output number of channels.
        dilations (tuple[int]): Dilation rate of each layer.
    """

    def __init__(self, in_channel, out_channel, dilations):
        super().__init__()
        for dilation in dilations:
            self.append(ConvModule(
                in_channel,
                out_channel, 
                1 if dilation == 1 else 3, 
                dilation=dilation,
                padding=0 if dilation == 1 else dilation, 
                bias=False))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


class ASPPHead(nn.Module):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`.

    Args:
        in_channel (int): Input number of channels. Default: 2048.
        out_channel (int): Output number of channels. Default: 512.
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, in_channel=2048, out_channel=512, dilations=(1, 6, 12, 18)):
        super().__init__()
        assert isinstance(dilations, (list, tuple))
        self.out_channel = out_channel
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(in_channel, out_channel, 1, bias=False),)
        self.aspp_modules = _ASPPModule(in_channel, out_channel, dilations)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * out_channel, out_channel, 1, bias=False)

    def _forward_impl(self, inputs):
        """Forward implementation."""
        x = inputs[-1]
        aspp_outs = [nn.functional.interpolate(
            self.image_pool(x), size=x.size()[2:], mode='bilinear')]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        return output

    def forward(self, inputs):
        """Forward function."""
        return self._forward_impl(inputs)


# class ASPPHeadPlus(ASPPHead):
#     """Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.

#     This head is a SIMPLIFIED implementation of `DeepLabV3+
#     <https://arxiv.org/abs/1802.02611>`.

#     Args:
#         in_channel (int): Input number of channels. Default: (256, 512, 1024, 2048).
#         out_channel (int): Output number of channels. Default: 512.
#         dilations (tuple[int]): Dilation rates for ASPP module.
#             Default: (1, 6, 12, 18).
#     """

#     def __init__(self,
#                  in_channels=(256, 512, 1024, 2048),
#                  out_channel=512,
#                  dilations=(1, 6, 12, 18)):
#         super().__init__(in_channels[-1], out_channel, dilations)
#         self.skip_proj = ConvModule(in_channels[0], out_channel, 1, bias=False)
#         self.out_proj = ConvModule(2 * out_channel, out_channel, 3, padding=1, bias=False)

#     def forward(self, inputs):
#         """Forward function."""
#         nonplus_out = nn.functional.interpolate(
#             self._forward_impl(inputs), size=inputs[0].size()[2:], mode='bilinear')
#         asppplus_out = torch.cat([self.skip_proj(inputs[0]), nonplus_out], dim=1)
#         output = self.out_proj(asppplus_out)
#         return output
