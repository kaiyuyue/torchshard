import warnings
import math
from typing import Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter

import torchshard

from .. import functional as F

from ...distributed import get_world_size, get_group, get_rank
from ...distributed import scatter, gather

from ...__init__ import set_attribute, _PARALLEL_DIM
from ...__init__ import TORCH_VERSION

# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')

class RegisterParallelDim(Module):
    """
    Helper class to register parallel dimension attribute to tensors.

    Args:
        dim (int): which dimension along to parallelize the tensor.

    Returns:
        x (Tensor): same as input tensor.
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(RegisterParallelDim, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            for _x in x:
                set_attribute(_x, self.dim)
        elif isinstance(x, Tensor):
            set_attribute(x, self.dim)
        else:
            raise ValueError

        return x

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class ParallelLinear(Module):
    """
    Parallel version of :class::`torch.nn.Linear`.

    Arguments are similar to :class:`torch.nn.Linear`. Extra arguments:
    Args:
        dim (int): which dimension along to parallelize layers.

    Returns:
        :class::`torch.Tensor`.
    """
    __constants__ = ['in_features', 'out_features', 'dim']
    in_features: int
    out_features: int
    weight: Tensor
    dim: Optional[int]

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dim: Optional[int] = None) -> None:
        super(ParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim

        self.weight = torch.empty((self.out_features, self.in_features))
        if bias:
            self.bias = torch.empty((self.out_features))
        else:
            self.bias = self.register_parameter('bias', None)

        self.reset_params()
        self.slice_params()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def reset_params(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def slice_params(self) -> None:
        # scatter weight and bias
        if self.dim == -1 or self.dim == 1:
            self.weight = scatter(self.weight, dim=0)
            if self.bias is not None:
                self.bias = scatter(self.bias, dim=0)

        if self.dim == 0:
            self.weight = scatter(self.weight, dim=1)

        # wrap into Parameter
        self.weight = Parameter(self.weight)
        if self.bias is not None:
            self.bias = Parameter(self.bias)

        # set parallel attr
        self._set_parameter_attr([self.weight, self.bias], self.dim)

    def forward(self, input: Tensor) -> Tensor:
        return F.parallel_linear(input, self.weight, self.bias, self.dim)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, dim={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.dim
        )

    @classmethod
    def _set_parameter_attr(cls, tensors, dim=None):
        if not isinstance(tensors, list):
            tensors = [tensors]
        for t in tensors:
            if t is None:
                continue
            set_attribute(t, dim)
            if isinstance(t, torch.nn.Parameter):
                # NOTE: Both setattr and __setattr__ doesn't work.
                set_attribute(t.data, dim)

    @classmethod
    def convert_parallel_linear(cls, module, dim=None):
        """
        Helper function to convert all :class::`torch.nn.Linear` layers in the model to
        :class::`torch.nn.ParallelLinear` layers.
        """
        if not dim in [None, 1, -1, 0]:
            raise ValueError(f"")
        module_output = module

        _ddp_params_and_buffers_to_ignore = []

        if isinstance(module, torch.nn.Linear):
            module_output = torchshard.nn.ParallelLinear(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None),
                dim=dim
            )
            with torch.no_grad():
                if dim is None:
                    module_output.weight = module.weight
                    if module.bias is not None:
                        module_output.bias = module.bias
                else:
                    if dim == -1 or dim == 1:
                        module_output.weight = Parameter(scatter(module.weight, dim=0))
                        if module.bias is not None:
                            module_output.bias = Parameter(scatter(module.bias, dim=0))
                    else:
                        module_output.weight = Parameter(scatter(module.weight, dim=1))
                        if module.bias is not None:
                            module_output.bias = module.bias

                # Above assignments will brush off the _PARALLEL_DIM. So we set it again.
                cls._set_parameter_attr(
                    [module_output.weight, module_output.bias], dim)

            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_parallel_linear(child, dim))

            # make DDP ignore parallel linear's weight and bias parameters
            module_output_child = getattr(module_output, name)
            if isinstance(module_output_child, torchshard.nn.ParallelLinear):
                if hasattr(module_output_child.weight, _PARALLEL_DIM):
                    parallel_dim = getattr(module_output_child.weight, _PARALLEL_DIM)
                    if parallel_dim is not None:
                        _ignore_name = f'{name}.weight'
                        _ddp_params_and_buffers_to_ignore.append(_ignore_name)

                if hasattr(module_output_child.bias, _PARALLEL_DIM):
                    parallel_dim = getattr(module_output_child.bias, _PARALLEL_DIM)
                    if parallel_dim is not None:
                        _ignore_name = f'{name}.bias'
                        _ddp_params_and_buffers_to_ignore.append(_ignore_name)

        del module

        # register for ignoring parallel linear in DDP
        setattr(module_output, '_ddp_params_and_buffers_to_ignore', _ddp_params_and_buffers_to_ignore)

        return module_output
