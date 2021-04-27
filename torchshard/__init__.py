# TorchShard Version
__version__ = "0.1.0"

# Python Version
import sys
if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is NOT supported by TorchShard.")

# PyTorch Version
import torch
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional, Union
__all__ = ['set_attribute']

# Define basic utilities
_PARALLEL_DIM = 'parallel_dim'

# NOTE: TorchShard has two order types:
# 0: [default] : input params order [in_feat_num, out_feat_num] = reverse(self.weight.shape)
# 1:           : input params order [in_feat_num, out_feat_num] =         self.weight.shape
_ORDER_ATTR = 'order_type'

def set_attribute(tensor: Union[Tensor, Parameter], value: Optional[int] = None, order_type: int = 0) -> None:
    """
    Inner helper function to set attributes to tensors.

    Attributes:
        - torchshard._PARALLEL_DIM (int): indicates which parallel manner produces the tensor. 
                                          It could be 
                                              0: produced by row parallel way
                                              1 or -1: produced by column way
                                              None: produced by non-parallel way
        - torchshard._ORDER_ATTR (int): indicates whether input parameters order is the same as the order of inner parameters (e.g., weight).
                                        Default is 0, which means they are in a reversed relation.
    """
    tensor.__setattr__(_PARALLEL_DIM, value)
    tensor.__setattr__(_ORDER_ATTR, order_type)

# Import most common subpackages
import torchshard.nn
import torchshard.distributed

# Import functions
from .serialization import (
    collect_state_dict,
    relocate_state_dict
)

from .overrides import register_ddp_parameters_to_ignore
from .overrides import (
    get_parallel_dim,
    register_parallel_dim,
    register_parallel_attribute
)

from ._tensor import slice
