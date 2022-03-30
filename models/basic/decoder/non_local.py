import einops
import math
import torch
import torch.nn as nn

from ..utils import ConvModule


class _NLAttentionModule(nn.Module):
    """Basic Non-local module.

    This module is proposed in "Non-local Neural Networks" `NLNet
    <https://arxiv.org/abs/1711.07971>`.

    Args:
        in_channel (int): Input number of channels.
        hidden_channel (int): Hidden number of channels.
        mode (str): The nonlocal mode. Options are 'gaussian', 'embedded_gaussian',
            'dot_product', 'concatenation'. Default: 'embedded_gaussian.'.
    """

    def __init__(self, in_channel, hidden_channel, mode='embedded_gaussian'):
        super().__init__()
        assert mode in ['gaussian', 'embedded_gaussian',
                        'dot_product', 'concatenation']
        self.mode = mode
        self.in_channel = in_channel
        self.d = hidden_channel
        self.theta = ConvModule(self.in_channel, self.d, 1,
                                conv=nn.Conv2d if self.mode != 'gaussian' else None,
                                bn=None, act=None, bias=False)
        self.phi = ConvModule(self.in_channel, self.d, 1,
                              conv=nn.Conv2d if self.mode != 'gaussian' else None,
                              bn=None, act=None, bias=False)
        self.g = ConvModule(self.in_channel, self.d, 1, bias=False)
        if self.mode == 'concatenation':
            self.concat_project = ConvModule(self.d * 2, 1, 1, bias=False)
        self.conv_out = ConvModule(self.d, self.in_channel, 1, bias=False)

    def _gaussian(self, q, k):
        pairwise_weight = torch.bmm(q, k.transpose(1, 2))
        return nn.functional.softmax(pairwise_weight, dim=1)

    def _embedded_gaussian(self, q, k):
        pairwise_weight = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d)
        return nn.functional.softmax(pairwise_weight, dim=1)

    def _dot_product(self, q, k):
        pairwise_weight = torch.bmm(q, k.transpose(1, 2)) / self.d
        return pairwise_weight

    def _concatenation(self, q, k):
        pairwise_weight = self.concat_project(torch.cat([q, k], dim=1))
        return pairwise_weight.squeeze(1)

    def forward(self, q, k, v):
        """Forward function.

        Args:
            q (query): Tensor [B, C, H, W]
            k (key): Tensor [B, C, H, W]
            v (value): Tensor [B, C, H, W]

        Returns:
            attn: Attention map [B, 1, H, W]
        """
        residual = v

        q = einops.rearrange(self.theta(q), 'B d H W -> B (H W) d') \
            if self.mode != 'concatenation' else self.theta(q)
        k = einops.rearrange(self.phi(k), 'B d H W -> B (H W) d') \
            if self.mode != 'concatenation' else self.phi(k)
        v = einops.rearrange(self.g(v), 'B d H W -> B (H W) d')

        pairwise_func = getattr(self, '_' + self.mode)
        pairwise_weight = pairwise_func(q, k)

        output = einops.rearrange(torch.bmm(pairwise_weight, v),
                                  'B (H W) d -> B d H W', H=residual.size(-2), W=residual.size(-1))
        output = residual + self.conv_out(output)

        return output


class NLHead(nn.Module):
    """Non-local Neural Networks.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        in_channel (int): Input number of channels. Default: 2048.
        hidden_channel (int): Hidden number of channels. Default: 512.
        reduction (int): Reduction factor of projection transform. Default: 2.
        mode (str): The nonlocal mode. Options are 'gaussian', 'embedded_gaussian',
            'dot_product', 'concatenation'. Default: 'embedded_gaussian.'.
    """

    def __init__(self, in_channel=2048, out_channel=512, reduction=2, mode='embedded_gaussian'):
        super().__init__()
        self.out_channel = out_channel
        self.conv_in = ConvModule(
            in_channel, out_channel, 3, padding=1, bias=False)
        self.nl_block = _NLAttentionModule(
            out_channel, max(out_channel // reduction, 1), mode)
        self.conv_out = ConvModule(
            out_channel, out_channel, 3, padding=1, bias=False)

    def forward(self, inputs):
        """Forward function."""
        x = inputs[-1]
        x = self.conv_in(x)
        x = self.nl_block(x, x, x)
        output = self.conv_out(x)
        return output
