from typing import Union, Dict, Set, List, Any, Callable, Iterable, Type, Optional

import torch

from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Module

from .__init__ import TORCH_VERSION
from .__init__ import set_attribute, _PARALLEL_DIM


__all__ = ['register_ddp_parameters_to_ignore']


def register_ddp_parameters_to_ignore(module: Module) -> List:
    """
    Automatically detect out TorchShard layers and
    register them into DistributedDataParallel().parameters_to_ignore.

    Note:
        This function comes with PyTorch >= 1.8.0.
    """
    if TORCH_VERSION < (1, 8) and dim != None:
        raise ValueError(f"torchshard.register_ddp_parameters_to_ignore function" +
                         " only works in an expected manner with PyTorch >= 1.8.0." +
                         " But got {torch.__version__}.")

    # refer to :class::`torch.nn.parallel.DistributedDataParallel`
    _ddp_params_and_buffers_to_ignore = []

    for name, param in module.named_parameters():
        if hasattr(param, _PARALLEL_DIM):
            parallel_dim = getattr(param, _PARALLEL_DIM)
            if parallel_dim is not None:
                _ignore_name = name
                _ddp_params_and_buffers_to_ignore.append(_ignore_name)

    # register
    setattr(module, '_ddp_params_and_buffers_to_ignore', _ddp_params_and_buffers_to_ignore)

    return _ddp_params_and_buffers_to_ignore

def register_parallel_attribute(param: Union[Tensor, Parameter], value: Optional[int] = None, order_type: int = 0) -> None:
    """
    Helper function to resgister all the parallel attributes to the input tensor.
    """
    set_attribute(param, value, order_type)

def register_parallel_dim(tensor: Union[Tensor, Parameter], value: Optional[int] = None) -> None:
    """
    Helper function to register parallel dimension attribute to the input tensor.
    """
    set_attribute(tensor, value)

def get_parallel_dim(tensor: Union[Tensor, Parameter]) -> Optional[int]:
    """
    Helper to return the parallel dimension number of the input tensor.
    """
    if hasattr(tensor, _PARALLEL_DIM):
        return getattr(tensor, _PARALLEL_DIM)
    else:
        return None
