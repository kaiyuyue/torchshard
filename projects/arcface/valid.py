import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm

import torchshard as ts

parser = argparse.ArgumentParser(description='PyTorch ArcFace Validation')
parser.add_argument('--imgs-root', default=None, type=str,
                    help='path to valid dataset')
parser.add_argument('--pair-list', default=None, type=str,
                    help='pair list path for validation')
parser.add_argument('-a', '--arch', metavar='ARCH', default='iresnet18',
                    help='model architecture: (default: iresnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    # Here we use non-parallel model version on purpose. 
    # It can be also parallelized with torchshard to speed up inference.
    import iresnet
    print("=> creating model '{}'".format(args.arch))
    model = iresnet.__dict__[args.arch](
        num_classes=85744,
        enable_model_parallel=False,
        parallel_dim=None,
        eval_mode=True # for skipping the last layer
    )

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> load {} successfully".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data loading code
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    from utils import PairDataset
    valid_dataset = PairDataset(
        args.imgs_root, args.pair_list,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    valid_sampler = None
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=(valid_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=valid_sampler)

    validate(valid_loader, model, args)
    return

def validate(valid_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')

    # switch to evaluate mode
    model.eval()

    print("=> start to extract features")
    feats = {}
    with torch.no_grad():
        end = time.time()
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images1, images2, target) in pbar:
            images1 = images1.cuda(non_blocking=True)
            images2 = images2.cuda(non_blocking=True)

            # cat
            images = torch.cat([images1, images2], dim=0)

            # compute output
            output = model(images)

            # norm
            output = F.normalize(output)

            # features
            embedding_feat1 = output[:images1.shape[0], :]
            embedding_feat2 = output[images2.shape[0]:, :]

            # to np
            embedding_feat1 = embedding_feat1.data.cpu().numpy()
            embedding_feat2 = embedding_feat2.data.cpu().numpy()
            target = target.data.cpu().numpy()

            # save
            if i == 0:
                feats.setdefault('feats1', embedding_feat1)
                feats.setdefault('feats2', embedding_feat2)
                feats.setdefault('target', target)
            else:
                feats['feats1'] = np.append(feats['feats1'], embedding_feat1, axis=0)
                feats['feats2'] = np.append(feats['feats2'], embedding_feat2, axis=0)
                feats['target'] = np.append(feats['target'], target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # report progress
            pbar.set_description(f"iter: {i} batch_time: {batch_time.val:.3f}")

    # reconstruct
    feats1 = np.asarray(feats['feats1'])
    feats2 = np.asarray(feats['feats2'])
    target = np.asarray(feats['target']).reshape(-1)

    # eval
    from utils import evaluate
    evaluate(feats1, feats2, target)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

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


if __name__ == '__main__':
    main()
