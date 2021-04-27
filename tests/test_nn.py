import unittest
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

class TestParallelCrossEntropy(unittest.TestCase):

    @staticmethod
    def run_test_parallel_cross_entropy(local_rank: int) -> None:
        set_seed(seed + local_rank)
        parallel_dim = -1

        x = torch.randn(batch_size, feats_size).cuda(local_rank)
        y = torch.randint(10, (batch_size,)).cuda(local_rank)

        dist.broadcast(x, 0)
        dist.broadcast(y, 0)

        model = LinearModel(feats_size, feats_size*2, bias=True, dim=parallel_dim).cuda(local_rank)
        raw_model = model.module if hasattr(model, "module") else model

        # align weight
        ts.nn.init.shard_init_helper_(
            torch.nn.init.kaiming_normal_,
            raw_model.layer2.weight,
            a=0, mode='fan_in', nonlinearity='relu'
        )
        master_weight = ts.distributed.gather(raw_model.layer2.weight.data, dim=0)
        raw_model.layer1.weight.data.copy_(master_weight)

        # align bias
        ts.nn.init.shard_init_helper_(
            torch.nn.init.constant_,
            raw_model.layer2.bias,
            val=0.5
        )
        master_bias = ts.distributed.gather(raw_model.layer2.bias.data, dim=0)
        raw_model.layer1.bias.data.copy_(master_bias)

        model.train()

        criterion1 = torch.nn.CrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)
        criterion2 = ts.nn.ParallelCrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)

        y1, y2 = model(x)

        # 1st assert: forward outputs
        gathered_y2 = ts.distributed.gather(y2)
        assertEqual(y1, gathered_y2, threshold=threshold)

        loss1 = criterion1(y1, y)
        loss2 = criterion2(y2, y)

        if loss_reduction_type == 'none':
            loss1 = loss1.sum()
            loss2 = loss2.sum()

        # 2nd assert: forward losses
        assertEqual(loss1, loss2, threshold=threshold)

        # 3rd assert: backward gradients
        loss1.backward()
        loss2.backward()

        assertEqual(
            raw_model.layer1.weight.grad,
            ts.distributed.gather(raw_model.layer2.weight.grad, dim=0),
            threshold=threshold
        )
        assertEqual(
            raw_model.layer1.bias.grad,
            ts.distributed.gather(raw_model.layer2.bias.grad, dim=0),
            threshold=threshold
        )

    @staticmethod
    def run_test_parallel_cross_entropy_within_ddp_mode(local_rank: int) -> None:
        set_seed(seed + local_rank)
        parallel_dim = None
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

        # switch mode
        raw_model.train()
        ddp_model.train()

        # 1st assert: forward outputs
        y1 = raw_model(x)
        y2 = ddp_model(x)
        assertEqual(y1, y2, threshold=threshold)

        # 2nd assert: forward losses
        raw_loss = raw_criterion(y1, y)
        ddp_loss = ddp_criterion(y2, y)
        assertEqual(raw_loss, ddp_loss, threshold=threshold)

        if loss_reduction_type == 'none':
            raw_loss = raw_loss.sum()
            ddp_loss = ddp_loss.sum()

        # 3rd assert: backward gradients
        raw_loss.backward()
        ddp_loss.backward()

        assertEqual(raw_model.module.fc.weight.grad, ddp_model.module.fc.weight.grad, threshold=threshold)
        assertEqual(raw_model.module.fc.bias.grad, ddp_model.module.fc.bias.grad, threshold=threshold)
        assertEqual(raw_model.module.conv.weight.grad, ddp_model.module.conv.weight.grad, threshold=threshold)
        assertEqual(raw_model.module.conv.bias.grad, ddp_model.module.conv.bias.grad, threshold=threshold)

    @staticmethod
    def run_test_parallel_cross_entropy_within_ddp_mode_and_row_parallel(local_rank: int) -> None:
        set_seed(seed + local_rank)
        parallel_dim = 0
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

        # switch mode
        raw_model.train()
        ddp_model.train()

        x = ts.distributed.gather(x, dim=0)
        y = ts.distributed.gather(y, dim=0)

        y1 = raw_model(x)
        y2 = ddp_model(x)

        # 1st assert: forward outputs
        assertEqual(y1, y2, threshold=threshold)

        raw_loss = raw_criterion(y1, y)
        ddp_loss = ddp_criterion(y2, y)

        if loss_reduction_type == 'none':
            raw_loss = raw_loss.sum()
            ddp_loss = ddp_loss.sum()

        # 2nd assert: forward losses
        assertEqual(raw_loss, ddp_loss, threshold=threshold)

        # 3rd assert: backward gradients
        raw_loss.backward()
        ddp_loss.backward()

        assertEqual(ts.distributed.gather(raw_model.module.fc.weight.grad, dim=-1), ddp_model.module.fc.weight.grad, threshold=threshold)
        assertEqual(raw_model.module.fc.bias.grad, ddp_model.module.fc.bias.grad, threshold=threshold)

        assertEqual(raw_model.module.conv.weight.grad, ddp_model.module.conv.weight.grad, threshold=threshold)
        assertEqual(raw_model.module.conv.bias.grad, ddp_model.module.conv.bias.grad, threshold=threshold)

    @staticmethod
    def run_test_parallel_cross_entropy_within_ddp_mode_and_col_parallel(local_rank: int) -> None:
        set_seed(seed + local_rank)
        parallel_dim = -1
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

        # switch mode
        raw_model.train()
        ddp_model.train()

        y1 = raw_model(x)
        y2 = ddp_model(x)

        # 1st assert: forward outputs
        gathered_y1 = ts.distributed.gather(y1, dim=1)
        gathered_y2 = ts.distributed.gather(y2, dim=0)
        assertEqual(gathered_y1, gathered_y2, threshold=threshold)

        # 2nd assert: forward losses
        gathered_y = ts.distributed.gather(y)
        raw_loss = raw_criterion(y1, gathered_y)
        ddp_loss = ddp_criterion(y2, y)

        if loss_reduction_type == 'none':
            raw_loss = raw_loss.sum()
            ddp_loss = ddp_loss.sum()

        # 3rd assert: backward gradients
        raw_loss.backward()
        ddp_loss.backward()

        if loss_reduction_type == 'mean':
            linear_w_grad = ddp_model.module.fc.weight.grad
            linear_b_grad = ddp_model.module.fc.bias.grad
        else:
            linear_w_grad = ts.distributed.reduce(ddp_model.module.fc.weight.grad)
            linear_b_grad = ts.distributed.reduce(ddp_model.module.fc.bias.grad)

        assertEqual(
            raw_model.module.fc.weight.grad,
            ts.distributed.scatter(linear_w_grad, dim=0),
            threshold=threshold
        )
        assertEqual(
            raw_model.module.fc.bias.grad,
            ts.distributed.scatter(linear_b_grad, dim=0),
            threshold=threshold
        )

        if loss_reduction_type == 'mean':
            conv_w_grad = ts.distributed.reduce(raw_model.module.conv.weight.grad)
            conv_b_grad = ts.distributed.reduce(raw_model.module.conv.bias.grad)
        else:
            conv_w_grad = raw_model.module.conv.weight.grad
            conv_b_grad = raw_model.module.conv.bias.grad

        assertEqual(conv_w_grad, ddp_model.module.conv.weight.grad, threshold=threshold)
        assertEqual(conv_b_grad, ddp_model.module.conv.bias.grad, threshold=threshold)


    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_parallel_cross_entropy(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_cross_entropy, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_parallel_cross_entropy_within_ddp_mode(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_cross_entropy_within_ddp_mode, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_parallel_cross_entropy_within_ddp_mode_and_row_parallel(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_cross_entropy_within_ddp_mode_and_row_parallel, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_parallel_cross_entropy_within_ddp_mode_and_col_parallel(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_cross_entropy_within_ddp_mode_and_col_parallel, ngpus),
            nprocs=ngpus
        )



class TestParallelLinearStack(unittest.TestCase):

    @staticmethod
    def run_test_parallel_linear_stack(local_rank: int) -> None:
        set_seed(seed + local_rank)
        x = torch.randn(batch_size, feats_size).cuda(local_rank)
        dist.broadcast(x, 0)

        model = LinearStackModel(feats_size, feats_size*2, bias=True).cuda(local_rank)
        raw_model = model.module if hasattr(model, "module") else model

        # align weight
        for idx, (m1, m2) in enumerate(zip(raw_model.module1.modules(), raw_model.module2.modules())):
            if idx == 0:
                continue

            # align weight and bias
            ts.nn.init.shard_init_helper_(
                torch.nn.init.xavier_normal_,
                m2.weight
            )
            ts.nn.init.shard_init_helper_(
                torch.nn.init.constant_,
                m2.bias,
                val=0.133
            )
            parallel_dim = getattr(m2.weight, ts._PARALLEL_DIM)

            if parallel_dim == None:
                master_weight = m2.weight.data
                master_bias = m2.bias.data
            elif parallel_dim == 0:
                master_weight = ts.distributed.gather(m2.weight.data, dim=1)
                master_bias = m2.bias.data
            elif parallel_dim == 1 or parallel_dim == -1:
                master_weight = ts.distributed.gather(m2.weight.data, dim=0)
                master_bias = ts.distributed.gather(m2.bias.data, dim=0)
            else:
                raise

            m1.weight.data.copy_(master_weight)
            m1.bias.data.copy_(master_bias)

        # forward
        model.train()
        y1, y2 = model(x)

        assertEqual(y1, y2, threshold=threshold)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_parallel_linear_stack(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_linear_stack, ngpus),
            nprocs=ngpus
        )


class TestParallelLinear(unittest.TestCase):

    @staticmethod
    def run_test_raw_parallel_linear(local_rank):
        set_seed(seed + local_rank)
        parallel_dim = None

        x = torch.randn(batch_size, feats_size).cuda(local_rank)
        dist.broadcast(x, 0)

        model = LinearModel(feats_size, feats_size*2, bias=True, dim=parallel_dim).cuda(local_rank)
        model = model.module if hasattr(model, "module") else model

        # align weight
        ts.nn.init.shard_init_helper_(
            torch.nn.init.kaiming_normal_,
            model.layer2.weight,
        )
        model.layer1.weight.data.copy_(model.layer2.weight.data)

        # align bias
        ts.nn.init.shard_init_helper_(
            torch.nn.init.constant_,
            model.layer2.bias,
            val=0.1
        )
        model.layer1.bias.data.copy_(model.layer2.bias.data)

        # forward
        model.train()

        y1, y2 = model(x)

        assertEqual(y1, y2, threshold=threshold)

    @staticmethod
    def run_test_row_parallel_linear(local_rank):
        set_seed(seed + local_rank)
        parallel_dim = 0

        x = torch.randn(batch_size, feats_size).cuda(local_rank)
        dist.broadcast(x, 0)

        model = LinearModel(feats_size, feats_size*2, bias=True, dim=parallel_dim).cuda(local_rank)

        # align weight
        ts.nn.init.shard_init_helper_(
            torch.nn.init.kaiming_normal_,
            model.layer2.weight,
            a=0, mode='fan_in', nonlinearity='leaky_relu'
        )
        master_weight = ts.distributed.gather(model.layer2.weight.data, dim=-1)
        model.layer1.weight.data.copy_(master_weight)

        # align bias
        ts.nn.init.shard_init_helper_(
            torch.nn.init.constant_,
            model.layer2.bias,
            val=0.333
        )
        model.layer1.bias.data.copy_(model.layer2.bias.data)

        # forward
        model.train()

        y1, y2 = model(x)

        assertEqual(y1, y2, threshold=threshold)

    @staticmethod
    def run_test_col_parallel_linear(local_rank):
        set_seed(seed + local_rank)
        parallel_dim = -1

        x = torch.randn(batch_size, feats_size).cuda(local_rank)
        dist.broadcast(x, 0)

        model = LinearModel(feats_size, feats_size*2, bias=True, dim=parallel_dim).cuda(local_rank)
        model = model.module if hasattr(model, "module") else model

        # align weight
        ts.nn.init.shard_init_helper_(
            torch.nn.init.kaiming_normal_,
            model.layer2.weight,
            a=0, mode='fan_in', nonlinearity='leaky_relu'
        )
        master_weight = ts.distributed.gather(model.layer2.weight.data, dim=0)
        model.layer1.weight.data.copy_(master_weight)

        # align bias
        ts.nn.init.shard_init_helper_(
            torch.nn.init.constant_,
            model.layer2.bias,
            val=0.5
        )
        master_bias = ts.distributed.gather(model.layer2.bias.data, dim=0)
        model.layer1.bias.data.copy_(master_bias)

        # forward
        model.train()

        y1, y2 = model(x)
        y2 = ts.distributed.gather(y2)

        assertEqual(y1, y2, threshold=threshold)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_col_parallel_linear(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_col_parallel_linear, ngpus),
            nprocs=ngpus
        )

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_raw_parallel_linear(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_raw_parallel_linear, ngpus),
            nprocs=ngpus
        )

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_row_parallel_linear(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_row_parallel_linear, ngpus),
            nprocs=ngpus
        )


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    unittest.main()
