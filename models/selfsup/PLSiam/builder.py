import einops
import os
import torch
import torch.nn as nn

from models.basic.encoder.resnet import seg_resnet50
from models.basic.utils import ConvMLP

from models.basic.decoder.fcn import MoCoFCN
# from models.basic.decoder.pspnet import PSPHead
# from models.basic.decoder.aspp import ASPPHead


class PatchLevelEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.backbone = seg_resnet50()

        self.decode_head = MoCoFCN()
        # self.decode_head = PSPHead()
        # self.decode_head = ASPPHead()
        
        self.fc = ConvMLP(256, output_dim)

    def forward(self, x):
        encoder_outputs = self.backbone(x)
        decoder_output = self.decode_head(encoder_outputs)
        output = self.fc(decoder_output)
        return output


class PixSiam(nn.Module):
    def __init__(self, base_encoder, pretrained=None, dim=256):
        super().__init__()
        self.encoder = base_encoder(dim)
        self.predictor = ConvMLP(dim, dim)

        if pretrained:
            if os.path.isfile(pretrained):
                print(f"=> loading pretrained model '{pretrained}'")
                state_dict = torch.load(pretrained)
                load_state_dict_result = self.encoder.load_state_dict(state_dict, strict=False)
                print(load_state_dict_result)
                # sperate learning rate
                self.pretrained_params, self.rest_params = [], []
                for name, p in self.encoder.named_parameters():
                    if name in load_state_dict_result.missing_keys:
                        self.rest_params.append(p)
                    else:
                        self.pretrained_params.append(p)
                for name, p in self.predictor.named_parameters():
                    self.rest_params.append(p)
            else:
                print(f"=> no pretrained model found at '{pretrained}'")

    @staticmethod
    def _cos_sim(a, b, mask_a):
        a = nn.functional.normalize(einops.rearrange(a, 'b c h w -> b (h w) c'), dim=-1)
        b = nn.functional.normalize(einops.rearrange(b, 'b c h w -> b (h w) c'), dim=-1)
        cos_sims = (torch.bmm(a, b.transpose(1, 2)) * mask_a).sum(dim=-1)
        avg_cos_sim = cos_sims.sum() / len(cos_sims.nonzero())
        return avg_cos_sim

    def forward(self, im_q, im_k, mask_q):
        q0, k0 = self.encoder(im_q), self.encoder(im_k)
        q1, k1 = self.predictor(q0), self.predictor(k0)
        normalized_mask_q = nn.functional.normalize(mask_q, p=1, dim=-1)
        normalized_mask_k = nn.functional.normalize(mask_q.transpose(1, 2), p=1, dim=-1)
        # compute loss
        loss = - (self._cos_sim(q1, k0.detach(), normalized_mask_q) + 
                  self._cos_sim(k1, q0.detach(), normalized_mask_k)) / 2
        return loss