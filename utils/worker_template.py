from abc import ABCMeta, abstractmethod
import builtins
import math
import os
import sys
import time
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed

from .meters import AverageMeter, ProgressMeter


class BasicWorker(metaclass=ABCMeta):
    def __call__(self, gpu, ngpus_per_node, args):
        # check gpus
        args.gpu = gpu
        if args.gpu is not None:
            print(f"Use GPU: {args.gpu} for training")
        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu
            torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                                 world_size=args.world_size, rank=args.rank)

        # suppress printing from this on if not master
        if (args.multiprocessing_distributed and args.gpu != 0) or args.rank != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass
        else:
            self.log_file = os.path.join(args.work_dir,
                f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}-train.log")
            def print_and_log(*args):
                msg = ' '.join([str(arg) for arg in args])
                msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]: {msg}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                with open(self.log_file, 'a+') as f:
                    f.write(msg)
            builtins.print = print_and_log

        print("=> args\n" + str(args))

        self.ngpus_per_node = ngpus_per_node
        self.args = args
        self.run()

    def define_model(self):
        self.model = self._create_model()
        self._model_distributed()

    @abstractmethod
    def _create_model(self) -> nn.Module:
        pass

    def _model_distributed(self):
        if self.args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.args.gpu is not None:
                torch.cuda.set_device(self.args.gpu)
                self.model.cuda(self.args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.args.batch_size = int(
                    self.args.batch_size / self.ngpus_per_node)
                self.args.workers = int(
                    (self.args.workers + self.ngpus_per_node - 1) / self.ngpus_per_node)
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.args.gpu])
            else:
                self.model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu)
            self.model = self.model.cuda(self.args.gpu)
            # comment out the following line for debugging
            raise NotImplementedError(
                "Only DistributedDataParallel is supported.")
        else:
            # AllGather implementation (batch shuffle, queue update, etc.) in
            # this code only supports DistributedDataParallel.
            raise NotImplementedError(
                "Only DistributedDataParallel is supported.")

    def define_optimizer(self):
        # define optimizer
        if hasattr(self.model.module, 'pretrained_params') \
            and hasattr(self.model.module, 'rest_params'):
            self.optimizer = optim.SGD([
                {'name': 'pretrained', 'params': self.model.module.pretrained_params,
                 'lr': 0.1 * self.args.base_lr, 'momentum': self.args.momentum,
                 'weight_decay': self.args.weight_decay},
                {'name': 'rest', 'params': self.model.module.rest_params, 
                 'lr': self.args.base_lr, 'momentum': self.args.momentum, 
                 'weight_decay': self.args.weight_decay}
            ])
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(), self.args.base_lr,
                momentum=self.args.momentum, weight_decay=self.args.weight_decay
            )
        print("=> optimizer\n" + str(self.optimizer))

    def define_scaler(self):
        # define scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)
        if self.args.fp16:
            print("=> using mixed precision training\n" + str(self.scaler))

    def resume_if_specified(self):
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print(f"=> loading checkpoint '{self.args.resume}'")
                if self.args.gpu is None:
                    checkpoint = torch.load(self.args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = f'cuda:{self.args.gpu}'
                    checkpoint = torch.load(self.args.resume, map_location=loc)
                self.args.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scaler.load_state_dict(checkpoint['scaler'])
                print(f"=> loaded checkpoint '{self.args.resume}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{self.args.resume}'")

    def define_dataloader(self):
        train_dataset = self._create_dataset()
        if self.args.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            self.train_sampler = None

        self.train_loader = data.DataLoader(
            train_dataset, batch_size=self.args.batch_size // self.args.update_interval,
            shuffle=(self.train_sampler is None), num_workers=self.args.workers,
            pin_memory=True, sampler=self.train_sampler, drop_last=True)
        print("=> dataloader\n" + str(self.train_loader))

    @abstractmethod
    def _create_dataset(self) -> data.Dataset:
        pass

    def train(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        additional_meters = self._init_additional_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses] + list(additional_meters),
            prefix=f"Epoch: [{epoch+1}]")

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, data in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # forward
            loss, additional_info = self._forward(data)
            # record loss
            losses.update(loss.item())
            self._update_additional_meters(additional_meters, additional_info)
            # backward
            self._backward(loss, i)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == 0 or (i + 1) % self.args.print_freq == 0:
                progress.display(i + 1)

    def _init_additional_meters(self) -> Sequence[AverageMeter]:
        return []

    def _update_additional_meters(self, meters: Sequence[AverageMeter],
                                  info: Sequence[Union[torch.Tensor, int, float]]):
        if len(meters) != len(info):
            print('Warning, number of additional meters does not match the number of addtional info'
                  'Fail to update additional meters. Skipping this operation.')
        for meter, value in zip(meters, info):
            if isinstance(value, torch.Tensor):
                meter.update(value.item())
            elif isinstance(value, (int, float)):
                meter.update(value)

    @abstractmethod
    def _forward(self, data: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        pass

    def _backward(self, loss, i):
        # compute gradient
        (self.scaler.scale(loss) / self.args.update_interval).backward()
        # SGD update
        if (i + 1) % self.args.update_interval == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.zero_grad()

    def adjust_learning_rate(self, epoch):
        # cosine annealing
        base_lr = self.args.base_lr
        min_lr = 0.1 * self.args.base_lr
        cos_lr = min_lr + (base_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / (self.args.epochs-1)))
        print("=> adjust learning rate")
        for param_group in self.optimizer.param_groups:
            if param_group.get('name', None) == 'pretrained':
                pretrained_lr = base_lr + min_lr - cos_lr
                if pretrained_lr > cos_lr:
                    pretrained_lr = cos_lr
                param_group['lr'] = pretrained_lr
                print(param_group['name'], param_group['lr'])
            elif param_group.get('name', None) == 'rest':
                param_group['lr'] = cos_lr
                print(param_group['name'], param_group['lr'])
            else:
                param_group['lr'] = cos_lr
                print(param_group['lr'])

    def save_checkpoint(self, epoch):
        if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                         and self.args.rank % self.ngpus_per_node == 0):
            if (epoch + 1) % self.args.checkpoint_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': self.scaler.state_dict(),
                }, os.path.join(self.args.work_dir, f'checkpoint_{epoch + 1:03d}.pth'))

    def run(self):
        self.define_model()
        self.define_optimizer()
        self.define_scaler()
        self.resume_if_specified()
        self.define_dataloader()

        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            # train for one epoch
            self.adjust_learning_rate(epoch)
            self.train(epoch)
            self.save_checkpoint(epoch)
