import torch.nn as nn

from ..utils import ConvModule


# FIXME: High per pixel acc but LOW mIoU
class UNetHead(nn.Module):
    """U-Net: Convolutional Networks for Biomedical Image Segmentation

    This head is the custom implementation of
    `U-Net <https://arxiv.org/abs/1505.04597>`

    Args:
        in_channels (tuple[int]): Input number of channels.
            Default: (256, 512, 1024, 2048).
        out_channel (int): Output number of channels. Default: 512.
    """

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channel=512):
        super().__init__()
        assert len(in_channels) >= 4
        self.up_conv32s = nn.Sequential(
            ConvModule(in_channels[-1], in_channels[-1], 2,
                       conv=nn.ConvTranspose2d, stride=2, bias=False),
            ConvModule(in_channels[-1], in_channels[-2], 3, padding=1, bias=False))
        self.up_conv16s = nn.Sequential(
            ConvModule(in_channels[-2] * 2, in_channels[-2], 2,
                       conv=nn.ConvTranspose2d, stride=2, bias=False),
            ConvModule(in_channels[-2], in_channels[-3], 3, padding=1, bias=False))
        self.up_conv8s = nn.Sequential(
            ConvModule(in_channels[-3] * 2, in_channels[-3], 2,
                       conv=nn.ConvTranspose2d, stride=2, bias=False),
            ConvModule(in_channels[-3], in_channels[-4], 3, padding=1, bias=False))
        self.up_conv4s = nn.Sequential(
            ConvModule(in_channels[-4] * 2, in_channels[-4], 2,
                       conv=nn.ConvTranspose2d, stride=2, bias=False),
            ConvModule(in_channels[-4], out_channel, 3, padding=1, bias=False))

    def forward(self, inputs):
        """Forward function."""
        x = self.up_conv32s(inputs[-1])
        # in case the size of input images is not divisable by 32
        x = nn.functional.interpolate(
            x, size=inputs[-2].shape[2:], mode='bilinear')
        x = self.up_conv16s(torch.cat([inputs[-2], x], dim=1))
        # in case the size of input images is not divisable by 16
        x = nn.functional.interpolate(
            x, size=inputs[-3].shape[2:], mode='bilinear')
        x = self.up_conv8s(torch.cat([inputs[-3], x], dim=1))
        # in case the size of input images is not divisable by 8
        x = nn.functional.interpolate(
            x, size=inputs[-4].shape[2:], mode='bilinear')
        x = self.up_conv4s(torch.cat([inputs[-4], x], dim=1))
        # output stride 2
        return x
