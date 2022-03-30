import argparse


class ArchParser(argparse.ArgumentParser):
    def __init__(self, model_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('-a', '--arch', metavar='ARCH', choices=model_names,
                          help='model architecture: ' + ' | '.join(model_names))


class BasicParser(argparse.ArgumentParser):
    def __init__(self, description='PyTorch Segmentation Pretraining', **kwargs):
        super().__init__(description=description, **kwargs)
        self.add_argument('data_root', metavar='DIR', help='path to dataset')
        self.add_argument('--work-dir', default='./', metavar='DIR',
                          help='path to work directory (default: ./)')
        self.add_argument('--workers', default=32, type=int, metavar='N',
                          help='number of data loading workers (default: 32)')
        self.add_argument('--epochs', default=200, type=int, metavar='N',
                          help='number of total epochs to run (default: 200)')
        self.add_argument('--start-epoch', default=0, type=int, metavar='N',
                          help='manual epoch number (useful on restarts)')
        self.add_argument('--crop-size', default=512, type=int,
                          help='augmentation crop size (default: 512)')
        self.add_argument('--batch-size', default=256, type=int, metavar='N',
                          help='mini-batch size (default: 256), this is the total '
                          'batch size of all GPUs on the current node when '
                          'using Data Parallel or Distributed Data Parallel')
        self.add_argument('--base-lr', default=0.01, type=float, metavar='LR',
                          help='initial learning rate', dest='base_lr')
        self.add_argument('--momentum', default=0.9, type=float, metavar='M',
                          help='momentum of SGD solver')
        self.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                          help='weight decay (default: 1e-4)', dest='weight_decay')
        self.add_argument('--print-freq', default=10, type=int, metavar='N',
                          help='print frequency (default: 10 iters)')
        self.add_argument('--checkpoint-freq', default=10, type=int, metavar='N',
                          help='checkpoint frequency (default: 10 epochs)')
        self.add_argument('--resume', default='', type=str, metavar='PATH',
                          help='path to latest checkpoint (default: none)')
        self.add_argument('--pretrained', default='', type=str, metavar='PATH',
                          help='path to init checkpoint (default: none)')
        self.add_argument('--world-size', default=-1, type=int,
                          help='number of nodes for distributed training')
        self.add_argument('--rank', default=-1, type=int,
                          help='node rank for distributed training')
        self.add_argument('--dist-url', default='tcp://localhost:29500', type=str,
                          help='url used to set up distributed training')
        self.add_argument('--dist-backend', default='nccl',
                          type=str, help='distributed backend')
        self.add_argument('--seed', default=None, type=int,
                          help='seed for initializing training.')
        self.add_argument('--gpu', default=None,
                          type=int, help='GPU id to use.')
        self.add_argument('--multiprocessing-distributed', action='store_true',
                          help='Use multi-processing distributed training to launch '
                          'N processes per node, which has N GPUs. This is the '
                          'fastest way to use PyTorch for either single node or '
                          'multi node data parallel training')
        self.add_argument('--fp16', action='store_true',
                          help='mixed percision training')
        self.add_argument('--update-interval', default=1,
                          type=int, help='gradient update interval')
