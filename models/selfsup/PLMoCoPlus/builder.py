from math import log
import einops
import numpy as np
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


class PLMoCoPlus(nn.Module):
    """
    Build a MoCo model (modified to encode patch level representations)
    with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, pretrained=None, dim=128, K=65536, m=0.999, T=0.2):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.2)
        """
        super(PLMoCoPlus, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(dim)
        self.encoder_k = base_encoder(dim)

        if pretrained:
            if os.path.isfile(pretrained):
                print(f"=> loading pretrained model '{pretrained}'")
                state_dict = torch.load(pretrained)
                load_state_dict_result = self.encoder_q.load_state_dict(state_dict, strict=False)
                print(load_state_dict_result)
                # sperate learning rate
                self.pretrained_params, self.rest_params = [], []
                for name, p in self.encoder_q.named_parameters():
                    if name in load_state_dict_result.missing_keys:
                        self.rest_params.append(p)
                    else:
                        self.pretrained_params.append(p)
            else:
                print(f"=> no pretrained model found at '{pretrained}'")

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_dense", torch.randn(dim, K))
        self.queue_dense = nn.functional.normalize(self.queue_dense, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_dense_ptr", torch.zeros(1, dtype=torch.long))

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_dense(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_dense_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_dense[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_dense_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, mask_q):
        """
        Input:
            im_q: a batch of query images
            loc_q: a batch of query locations
            im_k: a batch of key images
            loc_k: a batch of key locations
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: BxCxHxW

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            # compute key features
            k = self.encoder_k(im_k)  # keys: BxCxHxW
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # reshape
        q = einops.rearrange(q, 'b c h w -> b (h w) c')
        k = einops.rearrange(k, 'b c h w -> b (h w) c')

        # ------------------------------ MoCo -------------------------------
        # normalize for cos sim
        q_instance = nn.functional.normalize(q.mean(dim=1), dim=-1)
        k_instance = nn.functional.normalize(k.mean(dim=1), dim=-1)
        # Einstein sum is more intuitive
        # positive logits: Bx1
        l_pos = torch.einsum('bc,bc->b', [q_instance, k_instance]).unsqueeze(-1)
        # negative logits: BxK
        l_neg = torch.einsum('bc,ck->bk', [q_instance, self.queue.clone().detach()])
        # logits: Bx(1+K), applied with temperature
        logits = torch.cat([l_pos, l_neg], dim=-1) / self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # compute loss
        loss = self.criterion(logits, labels)
        acc = accuracy(logits, labels.unsqueeze(1))[0]
        # dequeue and enqueue
        self._dequeue_and_enqueue(k_instance)
        #  ------------------------------------------------------------------

        # ----------------------------- PLMoCo ------------------------------
        # normalize for cos sim
        q_dense = nn.functional.normalize(q, dim=-1)
        k_dense = nn.functional.normalize(k, dim=-1)
        mask_q = nn.functional.normalize(mask_q, p=1, dim=-1)
        # Einstein sum is more intuitive
        # positive logits_dense: (BHW)x1
        l_pos_dense = einops.reduce(
            torch.bmm(q_dense, k_dense.transpose(1, 2)) * mask_q, 'b hw_q hw_k -> (b hw_q) 1', 'sum')
        # negative logits_dense: (BHW)xK
        l_neg_dense = torch.mm(
            einops.rearrange(q_dense, 'b hw_q c -> (b hw_q) c'), self.queue_dense.clone().detach())
        # logits_dense: (BHW)x(1+K)
        logits_dense = torch.cat([l_pos_dense, l_neg_dense], dim=-1) / self.T
        # labels: positive key indicators
        labels_dense = torch.zeros(logits_dense.shape[0], dtype=torch.long).cuda()
        # compute loss
        loss_dense = self.criterion(logits_dense, labels_dense)
        nonzero_idx = logits_dense[:, 0] > 0
        acc_dense = accuracy(logits_dense[nonzero_idx], labels_dense[nonzero_idx].unsqueeze(1))[0]
        # dequeue and enqueue
        k_dense = einops.rearrange(k_dense, 'b hw_k c -> (b hw_k) c')
        sample_idx = np.random.choice(k_dense.size(0), self.K // 64, replace=False)
        self._dequeue_and_enqueue_dense(k_dense[sample_idx])
        #  ------------------------------------------------------------------

        # combine losses
        combined_loss = (loss + loss_dense) / 2

        return combined_loss, loss, acc, loss_dense, acc_dense


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
