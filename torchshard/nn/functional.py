from typing import Optional, List

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.nn.parameter import Parameter

from . import _reduction as _Reduction

from ..__init__ import _PARALLEL_DIM
from ..__init__ import set_attribute

from ..distributed import scatter, reduce, gather, copy
from ..distributed import get_world_size, get_group_size
from ..distributed import get_group, get_rank

from ..utils import VocabUtility


def parallel_linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, dim: Optional[int] = None) -> Tensor:
    """
    Parallel version of :func::`torch.nn.functional.linear`.
    Applies a linear transformation to the incoming data: :math::`y = xA^T + b`.

    This operator supports :ref::`TensorFloat32<tf32_on_ampere>`.
    """
    assert weight.dim() == 2, 'weight must be two-dimensional, but got a {}D Tensor'.format(weight.dim())

    if dim == None:
        return F.linear(input, weight, bias=bias)
    elif dim == 0:
        return _row_parallel_linear(input, weight, bias=bias)
    elif dim == 1 or dim == -1:
        return _col_parallel_linear(input, weight, bias=bias)
    else:
        raise ValueError(f"parallel linear supports the 2D Tensor slice along the dimension of '0', '1', '-1' and 'None', but got {dim}")


def _row_parallel_linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    if hasattr(input, _PARALLEL_DIM) and getattr(input, _PARALLEL_DIM) in [-1, 1]:
        input = input
    else:
        input = scatter(input, dim=-1)

    output = F.linear(input, weight, bias=None)
    output = reduce(output)

    if bias is not None:
        output = torch.add(output, bias)

    set_attribute(input, -1)
    set_attribute(output, None)
    return output


def _col_parallel_linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    input = copy(input)

    output = F.linear(input, weight, bias=bias)

    set_attribute(input, None)
    set_attribute(output, -1)
    return output


def parallel_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean"
) -> Tensor:
    """
    Parallel version of :func::`torch.nn.functional.cross_entropy`.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    if hasattr(input, _PARALLEL_DIM):
        if getattr(input, _PARALLEL_DIM) in [-1, 1]:
            return ParallelCrossEntropy.apply(input, target, weight, None, ignore_index, None, reduction)
        else:
            return F.cross_entropy(input, target, weight, None, ignore_index, None, reduction)
    else:
        return F.cross_entropy(input, target, weight, None, ignore_index, None, reduction)


class ParallelCrossEntropy(torch.autograd.Function):
    """
    Column parallel version of :class::`torch.nn.CrossEntropy`.
    """
    @staticmethod
    def forward(
        ctx: "ParallelCrossEntropy",
        input: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean"
    ) -> Tensor:

        dim = input.dim()
        if dim != 2:
            raise ValueError(f'Expected 2 dimensional input, but got {dim}')

        if input.size(0) != target.size(0):
            raise ValueError(f'Expected input batch size ({input.size(0)}) to match target batch size ({target.size(0)}).')

        world_size = get_world_size()
        group = get_group()
        n, c = input.size()

        input_max = input.max(dim=-1, keepdim=True)[0]
        dist.all_reduce(
            input_max,
            op=dist.ReduceOp.MAX,
            group=group
        )
        input = input.sub(input_max)

        exp_sum = input.exp().sum(dim=-1, keepdim=True)
        dist.all_reduce(
            exp_sum,
            op=dist.ReduceOp.SUM,
            group=group
        )

        softmax = input.exp().div(exp_sum)

        one_hot = torch.zeros(
            (n, c * world_size),
            device=input.device,
            requires_grad=False
        )
        one_hot.scatter_(1, target.unsqueeze(dim=-1), 1)

        # scatter one-hot vectors into GPUs
        one_hot = scatter(one_hot, dim=-1)

        input = input.mul(one_hot).sum(dim=-1, keepdim=True)
        dist.all_reduce(
            input,
            op=dist.ReduceOp.SUM,
            group=group
        )

        out = torch.log(exp_sum).sub(input)

        # perform reduction
        reduction_enum = _Reduction.get_enum(reduction)
        if reduction == 'sum' and reduction_enum == 2:
            out = out.sum()
        elif reduction == 'mean' and reduction_enum == 1:
            out = out.mean()
        elif reduction == 'none' and reduction_enum == 0:
            pass
        else:
            raise ValueError(f"reduction must be in 'none' | 'sum' | 'mean', but got {reduction}. Default: 'mean'.")

        # TODO:
        # apply weight and reduction

        # save for backward
        ctx.save_for_backward(softmax, one_hot, torch.Tensor([reduction_enum]))

        return out

    @staticmethod
    def backward(
        ctx: "ParallelCrossEntropy",
        grad_output: Tensor
    ) -> Tensor:

        softmax, one_hot, reduction_enum, = ctx.saved_tensors

        grad_input = softmax.sub(one_hot)

        if reduction_enum == 1:
            grad_input.div_(softmax.shape[0])

        grad_input.mul_(grad_output)

        return grad_input, None, None, None, None, None, None


def parallel_embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None,
                       max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False, sparse: bool = False,
                       num_embeddings: int = None, dim: Optional[int] = None) -> Tensor:
    assert weight.dim() == 2, 'weight must be two-dimensional, but got a {}D Tensor'.format(weight.dim())

    if dim == None:
        return F.embedding(input, weight, padding_idx, max_norm,
                           norm_type, scale_grad_by_freq, sparse)
    elif dim == 0:
        return _row_parallel_embedding(input, weight, padding_idx, max_norm,
                                       norm_type, scale_grad_by_freq, sparse, num_embeddings)
    elif dim == 1 or dim == -1:
        return _col_parallel_embedding(input, weight, padding_idx, max_norm,
                                       norm_type, scale_grad_by_freq, sparse, num_embeddings)
    else:
        raise ValueError(f"parallel linear supports the 2D Tensor slice along the dimension of '0', '1', '-1' and 'None', but got {dim}")

def _row_parallel_embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None,
                            max_norm: Optional[float] = None, norm_type: float = 2.,
                            scale_grad_by_freq: bool = False, sparse: bool = False,
                            num_embeddings: Optional[int] = None) -> Tensor:
    # Divide the weight matrix along the vocaburaly dimension.
    vocab_start_index, vocab_end_index = \
        VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            get_rank(),
            get_group_size()
        )
    if get_group_size() > 1:
        # Build the mask.
        input_mask = (input < vocab_start_index) | \
                     (input >= vocab_end_index)
        # Mask the input.
        masked_input = input.clone() - vocab_start_index
        masked_input[input_mask] = 0
    else:
        masked_input = input

    # Get embeddings
    output = F.embedding(masked_input, weight, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse)

    # Mask the output embedding.
    if get_group_size() > 1:
        output[input_mask, :] = 0.0

    # Reduce across all the model parallel GPUs.
    output = reduce(output)
    return output


def _col_parallel_embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None,
                            max_norm: Optional[float] = None, norm_type: float = 2.,
                            scale_grad_by_freq: bool = False, sparse: bool = False,
                            num_embeddings: Optional[int] = None) -> Tensor:
    output = F.embedding(input, weight, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse)

    # Gather from all the model parallel GPUs.
    output = gather(output, dim=-1)
    return output

