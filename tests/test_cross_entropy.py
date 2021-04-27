from typing import Optional, List, Callable, Tuple

import torch
import random
import sys
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.parallel as paralle

import unittest
import torchshard as ts

from testing import IdentityLayer
from testing import dist_worker, assertEqual, set_seed
from testing import loss_reduction_type, threshold


def torch_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed, local_rank):
    set_seed(seed)

    identity = IdentityLayer((batch_size, seq_length, vocab_size),
                             scale=logits_scale).cuda(local_rank)
    logits = identity()
    target = torch.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size).cuda(local_rank)

    logits = logits.view(-1, logits.size()[-1])
    target = target.view(-1)

    loss = F.cross_entropy(logits, target, reduction=loss_reduction_type).view_as(target)
    if loss_reduction_type == 'none':
        loss = loss.sum()
    loss.backward()

    return loss, identity.weight.grad, logits, target


def torchshard_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed, local_rank):
    set_seed(seed)

    identity = IdentityLayer((batch_size, seq_length, vocab_size),
                             scale=logits_scale).cuda(local_rank)
    logits = identity()
    target = torch.LongTensor(
             size=(batch_size, seq_length)).random_(0, vocab_size).cuda(local_rank)

    logits = logits.view(-1, logits.size()[-1])
    target = target.view(-1)

    logits_parallel = ts.distributed.scatter(logits, dim=-1)

    loss = ts.nn.functional.parallel_cross_entropy(logits_parallel, target, reduction=loss_reduction_type)
    if loss_reduction_type == 'none':
        loss = loss.sum()
    loss.backward()

    return loss, identity.weight.grad, logits, target


class TestCrossEntropy(unittest.TestCase):

    @staticmethod
    def run_test_naive_cross_entropy(local_rank: int) -> None:

        # settings
        batch_size = 13
        seq_length = 17
        vocab_size_per_partition = 11
        logits_scale = 1000.0
        tensor_model_parallel_size = ts.distributed.get_world_size()
        vocab_size = vocab_size_per_partition * tensor_model_parallel_size
        seed = 1234

        loss_torch, grad_torch, logits_torch, target_torch = \
            torch_cross_entropy(batch_size, seq_length,
                                vocab_size, logits_scale,
                                seed, local_rank)

        loss_ts, grad_ts, logits_ts, target_ts = \
            torchshard_cross_entropy(batch_size, seq_length,
                                     vocab_size, logits_scale,
                                     seed, local_rank)

        assertEqual(logits_torch, logits_ts, threshold=threshold)
        assertEqual(target_torch, target_ts, threshold=threshold)

        assertEqual(loss_torch, loss_ts, threshold=threshold)
        assertEqual(grad_torch, grad_ts, threshold=threshold)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_naive_cross_entropy(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_naive_cross_entropy, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    unittest.main()
