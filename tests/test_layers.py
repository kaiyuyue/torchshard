from typing import Optional, List, Callable, Tuple

import torch
import random
import sys
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.parallel as paralle
import torch.distributed as dist

import unittest
import torchshard as ts

from testing import IdentityLayer, IdentityLayer2D
from testing import dist_worker, assertEqual, set_seed
from testing import loss_reduction_type, threshold


class TestLayers(unittest.TestCase):

    @staticmethod
    def run_test_row_parallel_linear(local_rank: int) -> None:
        # settings
        tensor_model_parallel_size = ts.distributed.get_world_size()
        seed = 12345
        set_seed(seed)
        input_size_coeff = 13
        input_size = input_size_coeff * tensor_model_parallel_size
        output_size_coeff = 17
        output_size = output_size_coeff * tensor_model_parallel_size
        batch_size = 7

        # layers
        identity_layer = IdentityLayer2D(batch_size, input_size).cuda(local_rank)
        linear_layer = ts.nn.ParallelLinear(input_size, output_size,
                                            bias=True, dim=0).cuda(local_rank)
        loss_weight = torch.randn([batch_size, output_size]).cuda(local_rank)

        # forward
        input_ = identity_layer()
        output = linear_layer(input_)
        loss = torch.mul(output, loss_weight).sum()

        # backward
        loss.backward()

        # values
        dLdY = loss_weight
        X = identity_layer.weight
        A = linear_layer.weight.cuda(local_rank)

        dLdA = torch.matmul(dLdY.t(), X)
        dLdb = torch.matmul(torch.ones(batch_size, 1).cuda(local_rank).t(), dLdY).view(-1)
        dLdX = torch.matmul(dLdY, A)

        my_dLdA = torch.split(dLdA, input_size_coeff,
                              dim=1)[local_rank].contiguous().clone()

        # assert
        assertEqual(my_dLdA, linear_layer.weight.grad, threshold=threshold)
        assertEqual(dLdb, linear_layer.bias.grad, threshold=threshold)

        my_dLdX = ts.distributed.gather(dLdX, dim=-1)
        assertEqual(my_dLdX, identity_layer.weight.grad, threshold=threshold)

    @staticmethod
    def run_test_naive_col_parallel_linear(local_rank: int) -> None:

        # settings
        tensor_model_parallel_size = ts.distributed.get_world_size()
        seed = 12345
        set_seed(seed)
        input_size_coeff = 13
        input_size = input_size_coeff * tensor_model_parallel_size
        output_size_coeff = 17
        output_size = output_size_coeff * tensor_model_parallel_size
        batch_size = 7

        # layers
        identity_layer = IdentityLayer2D(batch_size, input_size).cuda(local_rank)
        linear_layer = ts.nn.ParallelLinear(input_size, output_size, bias=True, dim=-1).cuda(local_rank)
        loss_weight = torch.randn([batch_size, output_size]).cuda(local_rank)

        # forward
        input_ = identity_layer()
        output = linear_layer(input_)
        output = ts.distributed.gather(output, dim=-1)
        loss = torch.mul(output, loss_weight).sum()

        # backward
        loss.backward()

        # values
        dLdY = loss_weight
        X = identity_layer.weight
        A = linear_layer.weight.cuda(local_rank)
        A = ts.distributed.gather(A, dim=0)

        dLdA = torch.matmul(dLdY.t(), X)
        dLdb = torch.matmul(torch.ones(batch_size, 1).cuda(local_rank).t(), dLdY).view(-1)
        dLdX = torch.matmul(dLdY, A)

        # assert
        my_dLdA = torch.split(dLdA, output_size_coeff,
                              dim=0)[local_rank].contiguous().clone()
        assertEqual(my_dLdA, linear_layer.weight.grad, threshold=threshold)

        my_dLdb = torch.split(dLdb, output_size_coeff,
                              dim=0)[local_rank].contiguous().clone()
        assertEqual(my_dLdb, linear_layer.bias.grad, threshold=threshold)

        assertEqual(dLdX, identity_layer.weight.grad, threshold=threshold)

    @staticmethod
    def run_test_parallel_embedding(local_rank: int) -> None:
        # settings
        tensor_model_parallel_size = ts.distributed.get_world_size()
        seed = 12345
        batch_size = 17
        seq_length = 23
        vocab_size = 48
        hidden_size = 16
        seed = 1236

        # test parallel_dim = None
        set_seed(123)
        input_data = torch.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size).cuda(local_rank)
        loss_weight = torch.randn([batch_size, seq_length, hidden_size]).cuda(local_rank)

        set_seed(seed)
        embedding_original = torch.nn.Embedding(vocab_size, hidden_size).cuda(local_rank)

        output = embedding_original(input_data)
        loss_original = torch.mul(output, loss_weight)
        loss_original.sum().backward()

        # align weight
        set_seed(2827)
        parallel_dim = None
        embedding_parallel = ts.nn.ParallelEmbedding(vocab_size, hidden_size, dim=parallel_dim).cuda(local_rank)
        embedding_parallel.weight.data.copy_(embedding_original.weight.data)

        output = embedding_parallel(input_data)
        loss_parallel = torch.mul(output, loss_weight)
        loss_parallel.sum().backward()

        assertEqual(loss_original, loss_parallel, threshold=threshold)
        assertEqual(embedding_original.weight.grad, embedding_parallel.weight.grad, threshold=threshold)

        # test parallel_dim = 0
        set_seed(1257)
        parallel_dim = 0
        embedding_parallel = ts.nn.ParallelEmbedding(vocab_size, hidden_size, dim=parallel_dim).cuda(local_rank)
        embedding_parallel.weight.data.copy_(
            ts.distributed.scatter(embedding_original.weight.data, dim=parallel_dim))

        output = embedding_parallel(input_data)
        loss_parallel = torch.mul(output, loss_weight)
        loss_parallel.sum().backward()

        assertEqual(loss_original, loss_parallel, threshold=threshold)
        assertEqual(
            ts.distributed.scatter(embedding_original.weight.grad, dim=parallel_dim),
            embedding_parallel.weight.grad, threshold=threshold)

        # test parallel_dim = 1
        parallel_dim = -1
        set_seed(2678)
        embedding_parallel = ts.nn.ParallelEmbedding(vocab_size, hidden_size, dim=parallel_dim).cuda(local_rank)
        embedding_parallel.weight.data.copy_(
            ts.distributed.scatter(embedding_original.weight.data, dim=parallel_dim))

        output = embedding_parallel(input_data)
        loss_parallel = torch.mul(output, loss_weight)
        loss_parallel.sum().backward()

        assertEqual(loss_original, loss_parallel, threshold=threshold)
        assertEqual(
            ts.distributed.scatter(embedding_original.weight.grad, dim=parallel_dim),
            embedding_parallel.weight.grad, threshold=threshold)


    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_naive_col_parallel_linear(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_naive_col_parallel_linear, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_naive_row_parallel_linear(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_row_parallel_linear, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_naive_parallel_embedding(self):
        ngpus = torch.cuda.device_count()
        mp.spawn(
            dist_worker,
            args=(self.run_test_parallel_embedding, ngpus),
            nprocs=ngpus
        )
        ts.distributed.destroy_process_group()


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    unittest.main()
