from typing import Optional, List, Callable, Tuple

import torch
import random
import sys
import torch.nn.functional as F
import torch.nn.parallel as parallel
import torch.multiprocessing as mp
import torch.nn.parallel as paralle
import torch.distributed as dist

import unittest
import torchshard as ts

from testing import IdentityLayer, IdentityLayer2D, IdentityLayer3D
from testing import CausalSelfAttention, ParallelCausalSelfAttention
from testing import MLP, ParallelMLP
from testing import dist_worker, assertEqual, set_seed
from testing import loss_reduction_type, threshold


class TestLayers(unittest.TestCase):

    @staticmethod
    def run_test_parallel_self_attention(local_rank: int) -> None:
        seed = 1235
        batch_size = 10
        sequence_length = 12
        vocab_size = 12
        hidden_size = 8
        num_att_heads_per_partition = 6
        hidden_size_per_att_head = 8
        dropout_prob = 0.0 # has to be zero

        tensor_model_parallel_size = ts.distributed.get_group_size()
        world_size = ts.distributed.get_world_size()
        set_seed(seed + local_rank)

        num_att_heads = num_att_heads_per_partition * world_size
        hidden_size = hidden_size_per_att_head * num_att_heads

        attention_mask = torch.randn(batch_size, 1, 1, sequence_length).cuda(local_rank)
        x = torch.randn(batch_size, sequence_length, hidden_size).cuda(local_rank)
        y = torch.randint(10, (batch_size,)).cuda(local_rank)

        raw_model = ParallelCausalSelfAttention(hidden_size, num_att_heads, dropout_prob).cuda(local_rank)
        raw_model = parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
        ts.register_ddp_parameters_to_ignore(raw_model)

        ddp_model = parallel.DistributedDataParallel(
            CausalSelfAttention(hidden_size, num_att_heads, dropout_prob).cuda(local_rank),
            device_ids=[local_rank]
        )

        raw_criterion = ts.nn.ParallelCrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)
        ddp_criterion = torch.nn.CrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)

        # align weight & bias
        for (on, op), (pn, pp) in zip(ddp_model.named_parameters(), raw_model.named_parameters()):
            parallel_dim = ts.get_parallel_dim(pp)
            if parallel_dim == None:
                pp.data.copy_(op.data)
            elif parallel_dim == 0:
                if len(pp.shape) == 2:
                    pp.data.copy_(ts.distributed.scatter(op.data, dim=-1))
                else:
                    pp.data.copy_(op.data)
            elif parallel_dim in [1, -1]:
                pp.data.copy_(ts.distributed.scatter(op.data, dim=0))

        # switch mode
        raw_model.train()
        ddp_model.train()

        attention_mask = ts.distributed.gather(attention_mask, dim=0)
        x = ts.distributed.gather(x, dim=0)
        y = ts.distributed.gather(y, dim=0)

        y1 = raw_model(x, attention_mask)
        y2 = ddp_model(x, attention_mask)

        # 1st assert: forward outputs
        assertEqual(y1, y2, threshold=threshold)

        raw_loss = raw_criterion(y1.view(batch_size*tensor_model_parallel_size, -1), y)
        ddp_loss = ddp_criterion(y2.view(batch_size*tensor_model_parallel_size, -1), y)

        if loss_reduction_type == 'none':
            raw_loss = raw_loss.sum()
            ddp_loss = ddp_loss.sum()

        # 2nd assert: forward losses
        assertEqual(raw_loss, ddp_loss, threshold=threshold)

        # 3rd assert: backward gradients
        raw_loss.backward()
        ddp_loss.backward()

        for (on, op), (pn, pp) in zip(ddp_model.named_parameters(), raw_model.named_parameters()):
            parallel_dim = ts.get_parallel_dim(pp)
            if parallel_dim == None:
                assertEqual(pp.grad, op.grad, threshold=threshold)
            elif parallel_dim == 0:
                if len(pp.shape) == 2:
                    pp_grad = ts.distributed.reduce(pp.grad)
                    op_grad = ts.distributed.reduce(ts.distributed.scatter(op.grad, dim=1))
                    assertEqual(pp_grad, op_grad, threshold=threshold)
                else:
                    assertEqual(pp.grad, op.grad, threshold=threshold)
            elif parallel_dim in [1, -1]:
                pp_grad = ts.distributed.reduce(pp.grad)
                op_grad = ts.distributed.reduce(ts.distributed.scatter(op.grad, dim=0))
                assertEqual(pp_grad, op_grad, threshold=threshold)

    @staticmethod
    def run_test_parallel_mlp(local_rank: int) -> None:
        # settings
        seed = 12345
        batch_size = 10
        sequence_length = 12
        vocab_size = 12
        hidden_size = 8
        dropout_prob = 0.0 # has to be zero

        tensor_model_parallel_size = ts.distributed.get_group_size()
        world_size = ts.distributed.get_world_size()

        # test parallel_dim = None
        set_seed(seed + local_rank)
        loss_weight = torch.randn(batch_size, sequence_length, hidden_size).cuda(local_rank)
        attention_mask = torch.randn(batch_size, 1, 1, sequence_length).cuda(local_rank)
        input_data = torch.randn(batch_size, sequence_length, hidden_size).cuda(local_rank)

        dist.broadcast(loss_weight, src=0)
        dist.broadcast(attention_mask, src=0)
        dist.broadcast(input_data, src=0)

        # build attention module
        original_model = MLP(hidden_size, dropout_prob).cuda(local_rank)
        parallel_model = MLP(hidden_size, dropout_prob).cuda(local_rank)
        # we convert nn.Linear() to ts.nn.ParallelLinear()
        cnt = 0
        for n, m in parallel_model.named_modules():
            if isinstance(m, torch.nn.Linear) and cnt == 0: # first linear layer
                parallel_model.mlp[0] = ts.nn.ParallelLinear.convert_parallel_linear(m, dim=1)
                cnt += 1
                continue
            if isinstance(m, torch.nn.Linear) and cnt == 1: # second linear layer
                parallel_model.mlp[2] = ts.nn.RegisterParallelDim(dim=-1)
                parallel_model.mlp[3] = ts.nn.ParallelLinear.convert_parallel_linear(m, dim=0)

        original_model = parallel.DistributedDataParallel(original_model, device_ids=[local_rank])
        parallel_model = parallel.DistributedDataParallel(parallel_model, device_ids=[local_rank])
        ts.register_ddp_parameters_to_ignore(parallel_model)

        # align weight & bias
        for (on, op), (pn, pp) in zip(original_model.named_parameters(), parallel_model.named_parameters()):
            parallel_dim = ts.get_parallel_dim(pp)
            if parallel_dim == None:
                pp.data.copy_(op.data)
            elif parallel_dim == 0:
                if len(pp.shape) == 2:
                    pp.data.copy_(ts.distributed.scatter(op.data, dim=-1))
                else:
                    pp.data.copy_(op.data)
            elif parallel_dim in [1, -1]:
                pp.data.copy_(ts.distributed.scatter(op.data, dim=0))

        # switch mode
        original_model.train()
        parallel_model.train()

        # assert: weight and bias
        for (on, op), (pn, pp) in zip(original_model.named_parameters(), parallel_model.named_parameters()):
            parallel_dim = ts.get_parallel_dim(pp)
            if parallel_dim == None:
                assertEqual(op, pp, threshold=threshold)
            elif parallel_dim == 0:
                if len(pp.shape) == 2:
                    assertEqual(pp, ts.distributed.scatter(op.data, dim=-1), threshold=threshold)
                else:
                    assertEqual(pp, op, threshold=threshold)
            elif parallel_dim in [1, -1]:
                assertEqual(pp, ts.distributed.scatter(op.data, dim=0), threshold=threshold)

        # 1st assert: forward outputs
        parallel_output = parallel_model(input_data)
        original_output = original_model(input_data)
        assertEqual(original_output, parallel_output, threshold=threshold)

        # 2nd assert: forward losses
        original_loss = torch.mul(original_output, loss_weight)
        parallel_loss = torch.mul(parallel_output, loss_weight)
        original_loss.sum().backward()
        parallel_loss.sum().backward()
        assertEqual(original_loss, parallel_loss, threshold=threshold)

        # 3rd assert: backward gradients
        for (on, op), (pn, pp) in zip(original_model.named_parameters(), parallel_model.named_parameters()):
            parallel_dim = ts.get_parallel_dim(pp)
            if parallel_dim == None:
                assertEqual(pp.grad, op.grad, threshold=threshold)
            elif parallel_dim == 0:
                if len(pp.shape) == 2:
                    pp_grad = ts.distributed.reduce(pp.grad)
                    op_grad = ts.distributed.reduce(ts.distributed.scatter(op.grad, dim=1))
                    assertEqual(pp_grad, op_grad, threshold=threshold)
                else:
                    assertEqual(pp.grad, op.grad, threshold=threshold)
            elif parallel_dim in [1, -1]:
                pp_grad = ts.distributed.reduce(pp.grad)
                op_grad = ts.distributed.reduce(ts.distributed.scatter(op.grad, dim=0))
                assertEqual(pp_grad, op_grad, threshold=threshold)

    @staticmethod
    def run_test_parallel_transformer_block(local_rank: int) -> None:
        seed = 123
        batch_size = 10
        sequence_length = 12
        vocab_size = 12
        hidden_size = 8
        num_att_heads_per_partition = 6
        hidden_size_per_att_head = 8
        dropout_prob = 0.0 # has to be zero

        tensor_model_parallel_size = ts.distributed.get_group_size()
        world_size = ts.distributed.get_world_size()
        set_seed(seed + local_rank)

        num_att_heads = num_att_heads_per_partition * world_size
        hidden_size = hidden_size_per_att_head * num_att_heads

        attention_mask = torch.randn(batch_size, 1, 1, sequence_length).cuda(local_rank)
        x = torch.randn(batch_size, sequence_length, hidden_size).cuda(local_rank)
        y = torch.randint(10, (batch_size,)).cuda(local_rank)

        raw_model = torch.nn.Sequential(
            ParallelCausalSelfAttention(hidden_size, num_att_heads, dropout_prob),
            ParallelMLP(hidden_size, dropout_prob)
        ).cuda(local_rank)
        raw_model = parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
        ts.register_ddp_parameters_to_ignore(raw_model)

        ddp_model = parallel.DistributedDataParallel(
            torch.nn.Sequential(
                CausalSelfAttention(hidden_size, num_att_heads, dropout_prob),
                MLP(hidden_size, dropout_prob)
            ).cuda(local_rank),
            device_ids=[local_rank]
        )

        raw_criterion = ts.nn.ParallelCrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)
        ddp_criterion = torch.nn.CrossEntropyLoss(reduction=loss_reduction_type).cuda(local_rank)

        # align weight & bias
        for (on, op), (pn, pp) in zip(ddp_model.named_parameters(), raw_model.named_parameters()):
            parallel_dim = ts.get_parallel_dim(pp)
            if parallel_dim == None:
                pp.data.copy_(op.data)
            elif parallel_dim == 0:
                if len(pp.shape) == 2:
                    pp.data.copy_(ts.distributed.scatter(op.data, dim=-1))
                else:
                    pp.data.copy_(op.data)
            elif parallel_dim in [1, -1]:
                pp.data.copy_(ts.distributed.scatter(op.data, dim=0))

        # switch mode
        raw_model.train()
        ddp_model.train()

        attention_mask = ts.distributed.gather(attention_mask, dim=0)
        x = ts.distributed.gather(x, dim=0)
        y = ts.distributed.gather(y, dim=0)

        y1 = raw_model.module[0](x, attention_mask)
        y1 = raw_model.module[1](y1)

        y2 = ddp_model.module[0](x, attention_mask)
        y2 = ddp_model.module[1](y2)

        # 1st assert: forward outputs
        assertEqual(y1, y2, threshold=threshold)

        raw_loss = raw_criterion(y1.view(batch_size*tensor_model_parallel_size, -1), y)
        ddp_loss = ddp_criterion(y2.view(batch_size*tensor_model_parallel_size, -1), y)

        if loss_reduction_type == 'none':
            raw_loss = raw_loss.sum()
            ddp_loss = ddp_loss.sum()

        # 2nd assert: forward losses
        assertEqual(raw_loss, ddp_loss, threshold=threshold)

        # 3rd assert: backward gradients
        raw_loss.backward()
        ddp_loss.backward()

        # 3rd assert: backward gradients
        for (on, op), (pn, pp) in zip(ddp_model.named_parameters(), raw_model.named_parameters()):
            parallel_dim = ts.get_parallel_dim(pp)
            if parallel_dim == None:
                assertEqual(pp.grad, op.grad, threshold=threshold)
            elif parallel_dim == 0:
                if len(pp.shape) == 2:
                    pp_grad = ts.distributed.reduce(pp.grad)
                    op_grad = ts.distributed.reduce(ts.distributed.scatter(op.grad, dim=-1))
                    assertEqual(pp_grad, op_grad, threshold=threshold)
                else:
                    assertEqual(pp.grad, op.grad, threshold=threshold)
            elif parallel_dim in [1, -1]:
                assertEqual(pp.grad, ts.distributed.scatter(op.grad, dim=0))


    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_parallel_self_attention(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_self_attention, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_parallel_mlp(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_mlp, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_parallel_transformer_block(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_transformer_block, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    unittest.main()
