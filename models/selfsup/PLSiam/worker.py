import torch
import torch.nn as nn
import torchvision.datasets as datasets

from utils.worker_template import BasicWorker
from .loader import TwoCropsTransform
from .builder import PixSiam, PatchLevelEncoder


class Worker(BasicWorker):
    def _create_model(self):
        model = PixSiam(PatchLevelEncoder, self.args.pretrained, self.args.dim)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print("=> model\n" + str(model))
        return model

    def _create_dataset(self):
        data_root = self.args.data_root
        transform_func = TwoCropsTransform(crop_size=self.args.crop_size)
        train_dataset = datasets.ImageFolder(data_root, transform=transform_func)
        print("=> dataset\n" + str(train_dataset))
        return train_dataset

    def _forward(self, data):
        # unpack data
        (images, mask_q), _ = data
        if self.args.gpu is not None:
            images[0] = images[0].cuda(self.args.gpu, non_blocking=True)
            images[1] = images[1].cuda(self.args.gpu, non_blocking=True)
            mask_q = mask_q.cuda(self.args.gpu, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            loss = self.model(im_q=images[0], im_k=images[1], mask_q=mask_q)
        # return loss and additional info (as list)
        return loss, []
