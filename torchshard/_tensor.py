from typing import Union, Dict, Set, List, Any, Callable, Iterable, Type, Optional

import torch

from torch import Tensor

from .__init__ import TORCH_VERSION
from .utils import divide

__all__ = ['slice']

def slice(tensor: Tensor, num_partitions: int, dim: Optional[int] = None, contiguous: bool = False) -> List[Tensor]:
    """
    Slice a tensor along the specific dimension.

    Args:
        tensor (Tensor): input tensor.
        num_partitions (int): how many shards will be sliced out.
        dim (int): which dimmension along to slice the tensor (default is last dim `-1`).
        contiguous (bool): return contiguous tensor chunks if `True` (default is `False`).

    Returns:
        List[Tensor, ...]. Same as `torch.split` return.
    """
    split_size = divide(tensor.shape[dim], num_partitions)
    tensor_list = torch.split(tensor, split_size, dim=dim)
    if contiguous:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list
