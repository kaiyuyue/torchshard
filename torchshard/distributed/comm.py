from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .core import (
    get_group,
    get_rank,
    get_world_size
)
from ..utils import divide

from ..__init__ import set_attribute


def _reduce(tensor: Tensor) -> Tensor:
    if get_world_size() == 1:
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=get_group(), async_op=False)

    set_attribute(tensor, None)
    return tensor


def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    if get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[dim], get_world_size())
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[get_rank()].contiguous()

    set_attribute(output, dim)
    return output


def _gather(tensor: Tensor, dim: int = -1) -> Tensor:
    if get_world_size() == 1:
        return tensor
    tensor_list = [torch.ones_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor, group=get_group(), async_op=False)

    output = torch.cat(tensor_list, dim=dim)

    set_attribute(output, dim)
    return output


def copy(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Copy.apply(input)
    return input

class Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Copy", input: Tensor) -> Tensor:
        return input

    @staticmethod
    def backward(ctx: "Copy", grad_output: Tensor) -> Tensor:
        return _reduce(grad_output)


def scatter(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Scatter.apply(input, dim)
    else:
        input = _split(input, dim=dim)
    return input


class Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Scatter", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx: "Scatter", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _gather(grad_output, dim=int(dim)), None


def reduce(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Reduce.apply(input)
    else:
        input = _reduce(input)
    return input

class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Reduce", input: Tensor) -> Tensor:
        return _reduce(input)

    @staticmethod
    def backward(ctx: "Reduce", grad_output: Tensor) -> Tensor:
        return grad_output


def gather(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Gather.apply(input, dim)
    else:
        input = _gather(input, dim=dim)
    return input

class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Gather", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx: "Gather", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _split(grad_output, dim=int(dim)), None

