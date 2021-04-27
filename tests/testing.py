import random
import math
import numpy as np
from typing import Optional, List, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.nn.parameter import Parameter

import torchshard as ts


# If you are going to contribute major changes, please make sure the changes
# pass all the tests with different reduction types.
loss_reduction_type = 'none' # none | mean | sum

# Threshold must be less than 1e-6
threshold = 1e-6

def dist_worker(local_rank: int, test_func: Callable, ngpus: int = 1) -> None:
    machine_nums = 1
    machine_rank = machine_nums - 1
    world_size = ngpus
    global_rank = machine_rank * ngpus + local_rank

    dist.init_process_group(
        backend='NCCL',
        init_method='tcp://127.0.0.1:23456',
        world_size=world_size,
        rank=global_rank
    )
    ts.distributed.init_process_group(
        group_size=world_size
    )

    test_func(local_rank)


def set_seed(seed: int = 42) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def barf() -> None:
    import pdb
    pdb.set_trace()


def assertEqual(tensor: Tensor, expected: Tensor, threshold: float = 0.001) -> None:
    # https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/test.py#L14-L20
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        for t, e in zip(tensor, expected):
            assertEqual(t, e)
    else:
        # print(tensor.shape, expected.shape)
        diff = (tensor - expected).abs().max()
        if diff > threshold:
            print(f'diff value = {diff:.16f} on rank {str(ts.distributed.get_rank()).zfill(3)}')
            barf()


def torch_version() -> Tuple[int, ...]:
    numbering = torch.__version__.split("+")[0].split(".")[:3]

    # Catch torch version if run against internal pre-releases, like `1.8.0a0fb`,
    if not numbering[2].isnumeric():
        # Two options here:
        # - either skip this version (minor number check is not relevant)
        # - or check that our codebase is not broken by this ongoing development.

        # Assuming that we're interested in the second usecase more than the first,
        # return the pre-release or dev numbering
        logging.warning(f"Pytorch pre-relase version {torch.__version__} - assuming intent to test it")
        numbering[2] = "0"

    return tuple(int(n) for n in numbering)


class IdentityLayer(torch.nn.Module):
    def __init__(self, size, scale=1.0):
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self):
        return self.weight

class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight

class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m, n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n, k))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


class LinearModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: Optional[bool] = None, dim: Optional[int] = None) -> None:
        super(LinearModel, self).__init__()
        self.layer1 = torch.nn.Linear(in_features, out_features, bias=bias)
        self.layer2 = ts.nn.ParallelLinear(in_features, out_features, bias=bias, dim=dim)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        y1 = self.layer1(x)
        y2 = self.layer2(x)
        return y1, y2


class LinearStackModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: Optional[bool] = None) -> None:
        super(LinearStackModel, self).__init__()

        combinations = [
            [None for _ in range(7)],      # 0 - naive
            [-1   for _ in range(7)],      # 1 - parallel linear in col dim
            [0    for _ in range(7)],      # 2 - parallel linear in row dim
            [None, 0, 0, 1, None, -1, 0],  # 3 - hybrid
            [0, 0, 1, None, -1, 0, 0, -1, None, -1, 1, 0, None, 0, -1] # hybrid2
        ]

        self.parallel_dims = combinations[4]
        self.features_nums = [in_features, out_features]

        self.module1 = torch.nn.ModuleList()
        self.module2 = torch.nn.ModuleList()

        i, j = 0, 1
        for idx in range(len(self.parallel_dims)):
            m1 = torch.nn.Linear(
                self.features_nums[i],
                self.features_nums[j],
                bias=bias
            )
            m2 = ts.nn.ParallelLinear(
                self.features_nums[i],
                self.features_nums[j],
                bias=bias,
                dim=self.parallel_dims[idx]
            )

            self.module1.append(m1)
            self.module2.append(m2)

            # swap in and out features
            i, j = j, i

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        y1 = x
        for m in self.module1:
            y1 = m(y1)

        y2 = x
        for i, (dim, m) in enumerate(zip(self.parallel_dims, self.module2)):
            y2 = m(y2)
            if hasattr(y2, ts._PARALLEL_DIM) and getattr(y2, ts._PARALLEL_DIM) in [-1, 1]:
                y2 = ts.distributed.gather(y2, dim=dim)
                setattr(y2, ts._PARALLEL_DIM, 'None')

        return y1, y2


class ConvLinearModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: Optional[bool] = None, dim: Optional[int] = None) -> None:
        super(ConvLinearModel, self).__init__()

        self.conv = torch.nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)
        self.fc = torch.nn.Linear(out_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        if isinstance(self.fc, ts.nn.ParallelLinear):
            if self.fc.dim in [-1, 1]:
                x = ts.distributed.gather(x, dim=0)
        x = self.fc(x)
        return x


def self_attention_forward(x, att_mask, mixed_kqv, proj, n_head, parallel=False):
    B, T, C = x.size()

    if parallel:
        hs = n_head // ts.distributed.get_group_size()
    else:
        hs = n_head

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    kqv = mixed_kqv(x).view(B, T, hs, 3 * (C // n_head))
    k, q, v = ts.slice(kqv, 3, dim=-1, contiguous=True)

    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(att_mask[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, -1) # re-assemble all head outputs side by side

    if parallel:
        ts.register_parallel_dim(y, -1)

    # output projection
    y = proj(y)

    return y


class CausalSelfAttention(nn.Module):
    """
    Borrow from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L37.
    """

    def __init__(self, n_embd: int, n_head: int, pdrop: float = 0) -> None:
        super(CausalSelfAttention, self).__init__()
        assert n_embd % n_head == 0

        # key, query, value projections for all heads
        self.kqv = nn.Linear(n_embd, n_embd * 3)
        # regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.p = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = n_head

    def forward(self, x, att_mask):
        return self_attention_forward(x, att_mask, self.kqv, self.p, self.n_head, parallel=False)


class ParallelCausalSelfAttention(nn.Module):
    """
    Borrow from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L37.
    """

    def __init__(self, n_embd: int, n_head: int, pdrop: float = 0) -> None:
        super(ParallelCausalSelfAttention, self).__init__()
        assert n_embd % n_head == 0

        # key, query, value projections for all heads
        self.kqv = ts.nn.ParallelLinear(n_embd, n_embd * 3, dim=1)

        # regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.p = ts.nn.ParallelLinear(n_embd, n_embd, dim=0)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = n_head

    def forward(self, x, att_mask):
        return self_attention_forward(x, att_mask, self.kqv, self.p, self.n_head, parallel=True)


class MLP(nn.Module):
    """Borrow from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L81
    """
    def __init__(self, n_embd: int, resid_pdrop: float = 0.) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.mlp(self.ln(x))
        return x


class ParallelMLP(nn.Module):
    """Borrow from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L81
    """
    def __init__(self, n_embd: int, resid_pdrop: float = 0.) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            ts.nn.ParallelLinear(n_embd, 4 * n_embd, dim=-1),
            nn.GELU(),
            ts.nn.RegisterParallelDim(-1),
            ts.nn.ParallelLinear(4 * n_embd, n_embd, dim=0),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.mlp(self.ln(x))
        return x

