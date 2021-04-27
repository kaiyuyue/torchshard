import argparse
import os
import random
import shutil
import time
import warnings
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets

import torchshard as ts

parser = argparse.ArgumentParser(description='PyTorch minGPT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8007', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# args for torchshard
parser.add_argument('--enable-model-parallel', action='store_true',
                    help='enable training with torchshard')
parser.add_argument('--num-classes', default=1000, type=int,
                    help='number of classes')
parser.add_argument('--enable-one-data-group', action='store_true',
                    help='1-way data parallel and N-way model parallel')

# args for amp and ZeRO
parser.add_argument('--enable-amp-mode', action='store_true',
                    help='use mixed-precision training')
parser.add_argument('--enable-zero-optim', action='store_true',
                    help='use deepspeed-ZeRO optimizer')

# args for minGPT
parser.add_argument('--n-layer', default=12, type=int,
                    help='number of transformer layers')
parser.add_argument('--n-head', default=8, type=int,
                    help='number of transformer heads')
parser.add_argument('--n-embd', default=256, type=int,
                    help='transformer embedding size')


# counter used for learning rate decay
tokens = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global tokens
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        # always init model parallel processes and groups
        # to use torchshard.distributed package
        ts.distributed.init_process_group(group_size=args.world_size)

    # pytorch helpfully makes it easy to download datasets, e.g. the common CIFAR-10 https://www.kaggle.com/c/cifar-10
    print("=> preparing CIFAR10")
    root = args.data
    train_data = datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
    valid_data = datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)

    # we use the original ipynb script to generate codebook `C` to
    # make sure that all the processes have an identity codebook in DDP
    location = lambda storage, loc: storage.cpu()
    C = torch.load('c.pt', map_location=location)

    from mingpt.utils import ImageDataset
    train_dataset = ImageDataset(train_data, C)

    # create model
    print("=> creating model")
    from mingpt.model import GPTConfig, GPT
    mconf = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        enable_model_parallel=args.enable_model_parallel,
    )
    model = GPT(mconf)
    if args.enable_model_parallel:
        # let DDP ignore parallel parameters
        ts.register_ddp_parameters_to_ignore(model)

    print("number of parameters: {:e}".format(
          sum(p.numel() for p in model.parameters())))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.enable_model_parallel:
        criterion = ts.nn.ParallelCrossEntropyLoss().cuda(args.gpu)

    # trainer config
    from mingpt.trainer import Trainer, TrainerConfig
    tokens_per_epoch = len(train_data) * train_dataset.block_size
    train_epochs = args.epochs # todo run a bigger model and longer, this is tiny

    # initialize a trainer instance and kick off training
    # we only use mingpt.TrainerConfig to set optimizer
    # besides, we add ZeRO optimizer to mingpt.model class
    tconf = TrainerConfig(
        learning_rate=args.lr,
        betas = (0.9, 0.95), weight_decay=0,
        lr_decay=True,
        warmup_tokens=tokens_per_epoch,
        final_tokens=train_epochs*tokens_per_epoch,
        ckpt_path='cifar10_model.pt',
        enable_zero_optim=args.enable_zero_optim
    )
    optimizer = model.module.configure_optimizers(tconf)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # raw_model = model.module if args.distributed else model
            raw_model = model.module
            if args.enable_model_parallel:
                raw_model.load_state_dict(ts.relocate_state_dict(raw_model, checkpoint))
            else:
                raw_model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.distributed:
        if args.enable_model_parallel:
            # * If args.enable_one_data_group = True, we train GPT with 
            #   1-way data parallel group and N-way model parallel on a single machine (8 GPUs).
            # * If args.enable_one_data_group = False, we train GPT with 
            #   N-way data parallel group and N-way model parallel on a single machine (8 GPUs).
            from mingpt.utils import ExtraDistributedSampler
            if args.enable_one_data_group:
                train_sampler = ExtraDistributedSampler(train_dataset)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, tokens, tconf, args)

        # ts.collect_state_dict() needs to see all the process groups
        state_dict = model.module.state_dict()
        if args.enable_model_parallel:
            state_dict = ts.collect_state_dict(model.module, state_dict)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            # save for every epoch
            torch.save(state_dict, tconf.ckpt_path)


def train(train_loader, model, criterion, optimizer, epoch, tokens, tconf, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr = AverageMeter('LR', ':e', moving_average=False)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, losses],
        prefix="Epoch: [{}]".format(epoch))

    # gradscaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp_mode)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        if args.enable_model_parallel and not args.enable_one_data_group:
            # You can comment this gather step for shard training,
            # then you may build more data parallel groups instead.
            images = ts.distributed.gather(images, dim=0)
            target = ts.distributed.gather(target, dim=0)

        with torch.cuda.amp.autocast(enabled=args.enable_amp_mode):
            # compute output
            output, _ = model(images)
            output = output.view(-1, output.size(-1))

            # register parallel dim
            ts.register_parallel_dim(output, -1)

            target = target.view(-1)
            loss = criterion(output, target)

        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), tconf.grad_norm_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # number of tokens processed this step (i.e. label is not -100)
        rank_tokens = (target >= 0).sum()
        if args.distributed and not args.enable_model_parallel:
            # sum reduce for all the processes
            rank_tokens = ts.distributed.reduce(rank_tokens)
        tokens += rank_tokens

        # calculate lr
        cur_lr = adjust_learning_rate(optimizer, epoch, tokens, tconf, args)
        lr.update(cur_lr)

        # logging
        if i % args.print_freq == 0:
            progress.display(i)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', moving_average=True):
        self.name = name
        self.fmt = fmt
        self.ma = moving_average
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.ma else self.val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, tokens, tconf, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    # https://github.com/karpathy/minGPT/blob/master/mingpt/trainer.py#L94
    if tconf.lr_decay:
        if tokens < tconf.warmup_tokens:
            # linear warmup
            lr_mult = float(tokens) / float(max(1, tconf.warmup_tokens))
        else:
            # cosine learning rate decay
            progress = float(tokens - tconf.warmup_tokens) / \
                       float(max(1, tconf.final_tokens - tconf.warmup_tokens))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        lr = args.lr * lr_mult
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
