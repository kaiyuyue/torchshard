import warnings
import math
from typing import Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter

import torchshard as ts


class ParallelLayer(Module):
    """
    Build a customized parallel layer.
    
    Steps:
        0. change class name
        1. add / delete input parameters optionally
        2. write the special processes for weight and bias in the self.procs_params()
        3. write the corresponding forward functions.
    """
    __constants__ = ['in_features', 'out_features', 'dim']
    in_features: int
    out_features: int
    weight: Tensor
    dim: Optional[int]

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dim: Optional[int] = None) -> None:
        super(ParallelLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim

        self.weight = torch.Tensor(self.out_features, self.in_features)
        if bias:
            self.bias = torch.Tensor(self.out_features)
        else:
            self.bias = self.register_parameter('bias', None)

        self.procs_params()
        self.reset_params()
        self.slice_params()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def procs_params(self) -> None:
        # process parameters and other stuff
        return

    def reset_params(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def slice_params(self) -> None:
        # scatter weight and bias
        if self.dim == -1 or self.dim == 1:
            self.weight = ts.distributed.scatter(self.weight, dim=0)
            if self.bias is not None:
                self.bias = ts.distributed.scatter(self.bias, dim=0)

        if self.dim == 0:
            self.weight = ts.distributed.scatter(self.weight, dim=1)

        # wrap into Parameter
        self.weight = Parameter(self.weight)
        if self.bias is not None:
            self.bias = Parameter(self.bias)
        
        # NOTE:
        # TorchShard has two order types:
        # 0: [default] : input params order [in_feat_num, out_feat_num] = reverse(self.weight.shape)
        # 1:           : input params order [in_feat_num, out_feat_num] =         self.weight.shape
        # Please set corresponding order_type for your own layer.
        
        # set parallel attr
        ts.register_parallel_attribute(self.weight, self.dim, order_type=0)
        if self.bias is not None:
            ts.register_parallel_attribute(self.bias, self.dim, order_type=0)
          

    def forward(self, input: Tensor) -> Tensor:
        return F.parallel_linear(input, self.weight, self.bias, self.dim)

    def customized_forward(self, input: Tensor) -> Tensor:
        # If you would like to write your own forward
        # without torchshard.nn.functional.parallel_linear(),
        # please don't forget set parallel attributes for returns.

        # forward code ... modify condition flows according to order type
        if self.dim == 1 or self.dim == -1:
            otuput = _col_parallel_forward(input, self.weight, self.bias)
        elif self.dim == 0:
            output = _row_parallel_forward(input, self.weight, self.bias)
        else:
            output = _forward(input, self.weight, self.bias)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, dim={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.dim
        )


def _col_parallel_forward(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Parallel forward in column dimension.
    """
    input = ts.distributed.copy(input)

    # ...

    # set parallel attribute
    ts.register_parallel_dim(input, None)
    ts.register_parallel_dim(output, -1)
    return

def _row_parallel_forward(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Paralell forward in row dimension.
    """
    if hasattr(input, ts._PARALLEL_ATTR) and getattr(input, ts._PARALLEL_ATTR) in [-1, 1]:
        input = input
    else:
        input = scatter(input, dim=-1)

    # ...

    # set parallel attribute
    ts.register_parallel_dim(input, -1)
    ts.register_parallel_dim(output, None)
    return

def _forward(input: Tensor, weight: Tensor, bais: Optional[Tensor] = None) -> Tensor:
    """
    Naive forward function.
    """

    # ...

    # no need to set parallel attributes
    return
