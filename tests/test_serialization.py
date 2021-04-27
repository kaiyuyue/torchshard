import unittest
import shutil
import os
import copy
from typing import Optional, List, Callable, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel as parallel
from torch import Tensor

import torchshard as ts

from testing import dist_worker, assertEqual, set_seed
from testing import LinearModel, LinearStackModel, ConvLinearModel
from testing import loss_reduction_type, threshold


# global test configurations
batch_size = 3
feats_size = 8
seed = 12357
parallel_dim = -1
tmp_cache_root = '.cache' # for saving checkpoints

def _assert_equal(raw_model, ddp_model, parallel_dim):
    if parallel_dim in [-1, 1]:
        # align weight & bias
        raw_model.module.conv.weight.data.copy_(ddp_model.module.conv.weight.data)
        _weight = ts.distributed.scatter(ddp_model.module.fc.weight.data, dim=0)
        raw_model.module.fc.weight.data.copy_(_weight)

        raw_model.module.conv.bias.data.copy_(ddp_model.module.conv.bias.data)
        _bias = ts.distributed.scatter(ddp_model.module.fc.bias.data, dim=0)
        raw_model.module.fc.bias.data.copy_(_bias)

        # assert weight
        assertEqual(raw_model.module.conv.weight.data, ddp_model.module.conv.weight.data, threshold=threshold)
        assertEqual(raw_model.module.conv.bias.data, ddp_model.module.conv.bias.data, threshold=threshold)

        assertEqual(
            ts.distributed.gather(raw_model.module.fc.weight.data, dim=0),
            ddp_model.module.fc.weight.data,
            threshold=threshold
        )
        assertEqual(
            ts.distributed.gather(raw_model.module.fc.bias.data, dim=0),
            ddp_model.module.fc.bias.data,
            threshold=threshold
        )
    elif parallel_dim in [0]:
        # align weight & bias
        raw_model.module.conv.weight.data.copy_(ddp_model.module.conv.weight.data)
        raw_model.module.conv.bias.data.copy_(ddp_model.module.conv.bias.data)

        _weight = ts.distributed.scatter(ddp_model.module.fc.weight.data, dim=1)
        raw_model.module.fc.weight.data.copy_(_weight)
        raw_model.module.fc.bias.data.copy_(ddp_model.module.fc.bias.data)

        # assert weight
        assertEqual(raw_model.module.conv.weight.data, ddp_model.module.conv.weight.data, threshold=threshold)
        assertEqual(raw_model.module.conv.bias.data, ddp_model.module.conv.bias.data, threshold=threshold)

        assertEqual(
            ts.distributed.gather(raw_model.module.fc.weight.data, dim=1),
            ddp_model.module.fc.weight.data, threshold=threshold
        )
        assertEqual(raw_model.module.fc.bias.data, ddp_model.module.fc.bias.data, threshold=threshold)
    else:
        # align weight & bias
        raw_model.module.conv.weight.data.copy_(ddp_model.module.conv.weight.data)
        raw_model.module.conv.bias.data.copy_(ddp_model.module.conv.bias.data)

        raw_model.module.fc.weight.data.copy_(ddp_model.module.fc.weight.data)
        raw_model.module.fc.bias.data.copy_(ddp_model.module.fc.bias.data)

        # assert weight
        assertEqual(raw_model.module.conv.weight.data, ddp_model.module.conv.weight.data, threshold=threshold)
        assertEqual(raw_model.module.conv.bias.data, ddp_model.module.conv.bias.data, threshold=threshold)

        assertEqual(raw_model.module.fc.weight.data, ddp_model.module.fc.weight.data, threshold=threshold)
        assertEqual(raw_model.module.fc.bias.data, ddp_model.module.fc.bias.data, threshold=threshold)

class TestSerialization(unittest.TestCase):

    @staticmethod
    def run_test_save_model_state_dict(local_rank: int) -> None:
        set_seed(seed + local_rank)
        bias = True

        x = torch.randn(batch_size, 8, 1, 1).cuda(local_rank)
        y = torch.randint(10, (batch_size,)).cuda(local_rank)

        raw_model = ConvLinearModel(feats_size, feats_size*2, bias=bias, dim=parallel_dim).cuda(local_rank)
        # convert nn.Linear -> nn.ParallelLinear
        ts.nn.ParallelLinear.convert_parallel_linear(raw_model, dim=parallel_dim)
        raw_model = parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])

        ddp_model = parallel.DistributedDataParallel(
            ConvLinearModel(feats_size, feats_size*2, bias=bias, dim=parallel_dim).cuda(local_rank),
            device_ids=[local_rank]
        )

        raw_criterion = ts.nn.ParallelCrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)
        ddp_criterion = torch.nn.CrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)

        # assert
        _assert_equal(raw_model, ddp_model, parallel_dim)

        # switch mode
        raw_model.train()
        ddp_model.train()

        if parallel_dim == 0:
            x = ts.distributed.gather(x, dim=0)
            y = ts.distributed.gather(y, dim=0)

        # 1st assert: forward outputs
        y1 = raw_model(x)
        y2 = ddp_model(x)

        # 2nd assert: forward losses
        if parallel_dim in [-1, 1]:
            gathered_y = ts.distributed.gather(y)
        else:
            gathered_y = y
        raw_loss = raw_criterion(y1, gathered_y)
        ddp_loss = ddp_criterion(y2, y)

        if loss_reduction_type == 'none':
            raw_loss = raw_loss.sum()
            ddp_loss = ddp_loss.sum()

        # 3rd assert: backward gradients
        raw_loss.backward()
        ddp_loss.backward()

        # 4th assert: save model.state_dict()
        raw_model.eval()
        ddp_model.eval()

        # process state_dict
        raw_state_dict = raw_model.state_dict()
        raw_state_dict = ts.collect_state_dict(raw_model, raw_state_dict)
        ddp_state_dict = ddp_model.state_dict(keep_vars=False)

        # assert length
        assert len(raw_state_dict) == len(ddp_state_dict)

        # assert key and values
        for (raw_k, raw_t), (ddp_k, ddp_t) in zip(raw_state_dict.items(), ddp_state_dict.items()):
            assert raw_k == ddp_k
            assert type(raw_t) == type(ddp_t)
            assertEqual(raw_t, ddp_t)

        raw_pt_path = f'{tmp_cache_root}/raw.pt'
        ddp_pt_path = f'{tmp_cache_root}/ddp.pt'

        if local_rank == 0:
            os.makedirs(tmp_cache_root, exist_ok=True)

            torch.save({'state_dict': raw_state_dict}, raw_pt_path)
            torch.save({'state_dict': ddp_state_dict}, ddp_pt_path)

            # get pt file size
            raw_pt_size = os.path.getsize(raw_pt_path) / 1e6 # MB
            ddp_pt_size = os.path.getsize(ddp_pt_path) / 1e6 # MB
            # print(f'pt size: raw_pt_size = {raw_pt_size} MB, ddp_pt_size = {ddp_pt_size} MB')

        # delete
        del raw_state_dict
        del ddp_state_dict

    @staticmethod
    def run_test_load_model_state_dict(local_rank: int) -> None:
        set_seed(seed + local_rank)
        bias = True

        x = torch.randn(batch_size, 8, 1, 1).cuda(local_rank)
        y = torch.randint(10, (batch_size,)).cuda(local_rank)

        raw_model = ConvLinearModel(feats_size, feats_size*2, bias=bias, dim=parallel_dim).cuda(local_rank)
        # convert nn.Linear -> nn.ParallelLinear
        ts.nn.ParallelLinear.convert_parallel_linear(raw_model, dim=parallel_dim)
        raw_model = parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])

        ddp_model = parallel.DistributedDataParallel(
            ConvLinearModel(feats_size, feats_size*2, bias=bias, dim=parallel_dim).cuda(local_rank),
            device_ids=[local_rank]
        )

        raw_criterion = ts.nn.ParallelCrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)
        ddp_criterion = torch.nn.CrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)

        # switch mode
        raw_model.train()
        ddp_model.train()

        if parallel_dim == 0:
            x = ts.distributed.gather(x, dim=0)
            y = ts.distributed.gather(y, dim=0)

        # 1st assert: forward outputs
        y1 = raw_model(x)
        y2 = ddp_model(x)

        # 2nd assert: forward losses
        if parallel_dim in [-1, 1]:
            gathered_y = ts.distributed.gather(y)
        else:
            gathered_y = y
        raw_loss = raw_criterion(y1, gathered_y)
        ddp_loss = ddp_criterion(y2, y)

        if loss_reduction_type == 'none':
            raw_loss = raw_loss.sum()
            ddp_loss = ddp_loss.sum()

        # 3rd assert: backward gradients
        raw_loss.backward()
        ddp_loss.backward()

        # 4th assert: load model.state_dict()
        raw_pt_path = f'{tmp_cache_root}/raw.pt'
        ddp_pt_path = f'{tmp_cache_root}/ddp.pt'

        raw_state_dict = torch.load(raw_pt_path)['state_dict']
        ddp_state_dict = torch.load(ddp_pt_path)['state_dict']

        raw_state_dict = ts.relocate_state_dict(raw_model, raw_state_dict)

        # load
        raw_model.load_state_dict(raw_state_dict)
        ddp_model.load_state_dict(ddp_state_dict)

        _assert_equal(raw_model, ddp_model, parallel_dim)

        dist.barrier()
        if local_rank == 0:
            shutil.rmtree(tmp_cache_root)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_0_save_model_state_dict(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_save_model_state_dict, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_1_load_model_state_dict(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_load_model_state_dict, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()


