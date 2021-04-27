from typing import Any, Union

from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.parameter import Parameter

from .__init__ import _PARALLEL_DIM, _ORDER_ATTR
from .distributed import get_world_size, get_rank, get_group_size
from .distributed import gather
from .utils import check_divisibility, divide


def _record_parallel_layers(model: Module) -> 'OrderedDict[str, int]':
    mark = OrderedDict()
    for name, param in model.named_parameters():
        if hasattr(param, _PARALLEL_DIM):
            parallel_dim = param.__getattribute__(_PARALLEL_DIM)
            if hasattr(param, _ORDER_ATTR):
                order_type = param.__getattribute__(_ORDER_ATTR)
                if order_type == 1 and parallel_dim == 0:
                    parallel_dim = 1
                elif order_type == 1 and parallel_dim == 1:
                    parallel_dim = 0
                elif order_type == 1 and parallel_dim == -1:
                    parallel_dim = 0

            mark[name] = parallel_dim
    return mark

def collect_state_dict(model: Module, state_dict: 'OrderedDict[str, Tensor]', prefix: str = '') -> 'OrderedDict[str, Tensor]':
    """
    Helper function to collect states from all GPUs for save parallel layers.
    """
    if not isinstance(model, Module):
        raise ValueError

    destination = OrderedDict()
    mark = _record_parallel_layers(model)

    torch.cuda.synchronize()
    for name, t in state_dict.items():
        key = prefix + name
        if name in mark:
            parallel_dim = mark[name]
            if parallel_dim == 0:
                if len(t.shape) == 2:
                    t = gather(t, dim=1)
            elif parallel_dim == 1 or parallel_dim == -1:
                t = gather(t, dim=0)
            elif parallel_dim == None:
                pass
            else:
                raise ValueError

        destination[name] = t

    return destination

def relocate_state_dict(model: Module, state_dict: 'OrderedDict[str, Tensor]', prefix: str = '') -> 'OrderedDict[str, Tensor]':
    """
    Helper function to relocate states loaded from checkpoint to the shard parameters.
    """
    if not isinstance(model, Module):
        raise ValueError

    destination = OrderedDict()
    mark = _record_parallel_layers(model)

    torch.cuda.synchronize()
    for name, t in state_dict.items():
        key = prefix + name
        if name in mark:
            parallel_dim = mark[name]
            if parallel_dim == 0:
                if len(t.shape) == 2:
                    split_size = divide(t.shape[1], get_world_size())
                    tensor_list = torch.split(t, split_size, dim=1)
                    t = tensor_list[get_rank()].contiguous()
            elif parallel_dim == 1 or parallel_dim == -1:
                split_size = divide(t.shape[0], get_world_size())
                tensor_list = torch.split(t, split_size, dim=0)
                t = tensor_list[get_rank()].contiguous()
            elif parallel_dim == None:
                pass
            else:
                raise ValueError

        destination[name] = t

    return destination
