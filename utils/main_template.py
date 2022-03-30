import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp


class Main:
    def __init__(self, args, worker_func):
        self.args = args
        self.worker_func = worker_func

        # workdir
        os.makedirs(self.args.work_dir, exist_ok=True)

        cudnn.benchmark = True
        if self.args.seed is not None:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if self.args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')

        if self.args.dist_url == "env://" and self.args.world_size == -1:
            self.args.world_size = int(os.environ["WORLD_SIZE"])

        self.args.distributed = self.args.world_size > 1 or self.args.multiprocessing_distributed

    def run(self):
        ngpus_per_node = torch.cuda.device_count()
        if self.args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.args.world_size = ngpus_per_node * self.args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # worker process function
            mp.spawn(self.worker_func, nprocs=ngpus_per_node, args=(ngpus_per_node, self.args))
        else:
            # Simply call worker function
            self.worker_func(self.args.gpu, ngpus_per_node, self.args)

