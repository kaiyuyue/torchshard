import torch
import torch.distributed as dist

from typing import Any
from torch import Tensor

from ..distributed import get_world_size, get_group, get_rank
from ..distributed import scatter

from ..__init__ import _PARALLEL_DIM


def shard_init_helper_(init_method, tensor: Tensor, **kwargs: Any) -> None:
    """
    Helper function to initialize shard parameters.
    """
    if hasattr(tensor, _PARALLEL_DIM):
        local_rank = get_rank()
        group = get_group()
        world_size = get_world_size()

        parallel_dim = getattr(tensor, _PARALLEL_DIM)
        tensor_shape = list(map(int, tensor.shape))
        tensor_size = len(tensor_shape)

        # handle both weight and bias
        col_dim = tensor_size - 1
        row_dim = 0

        if parallel_dim == 0:
            tensor_shape[col_dim] *= world_size if tensor_size == 2 else 1
        elif parallel_dim == 1 or parallel_dim == -1:
            tensor_shape[row_dim] *= world_size
        elif parallel_dim == None:
            pass
        else:
            raise ValueError

        data = torch.empty(
            tensor_shape,
            dtype=torch.float,
            requires_grad=False
        ).cuda(local_rank)

        init_method(data, **kwargs)
        dist.broadcast(data, src=0)

        if parallel_dim == 0:
            data = scatter(data, dim=col_dim) if tensor_size == 2 else data
        elif parallel_dim == 1 or parallel_dim == -1:
            data = scatter(data, dim=row_dim)
        elif parallel_dim == None:
            pass
        else:
            raise ValueError()

        tensor.data.copy_(data)

        del data
    else:
        return init_method(tensor, **kwargs)

