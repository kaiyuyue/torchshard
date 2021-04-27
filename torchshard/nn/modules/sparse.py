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

from ...distributed import get_world_size, get_group_size
from ...distributed import get_group, get_rank
from ...distributed import scatter, gather, reduce

from ...__init__ import set_attribute, _PARALLEL_DIM, _ORDER_ATTR
from ...__init__ import TORCH_VERSION


class ParallelEmbedding(Module):
    """
    Parallel version of :class::`torch.nn.Embedding`.

    Arguments are similar to :class:`torch.nn.Embedding`. Extra arguments:
    Args:
        dim (int): which dimension along to parallelize layers.

    Returns:
        :class::`torch.Tensor`.
    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse', 'dim']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool
    dim: int

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, dim: Optional[int] = None) -> None:
        super(ParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.dim = dim
        if _weight is None:
            self.weight = torch.Tensor(num_embeddings, embedding_dim)
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = _weight
        self.slice_parameters()
        self.sparse = sparse

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def slice_parameters(self) -> None:
        # scatter weight
        if self.dim is not None:
            self.weight = scatter(self.weight, dim=self.dim)
        self.weight = Parameter(self.weight)
        self._set_parameter_attr(self.weight, self.dim)

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return F.parallel_embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
            self.num_embeddings, self.dim
        )

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        if self.dim is not None:
            s += ', dim={dim}'
        return s.format(**self.__dict__)

    @classmethod
    def _set_parameter_attr(cls, tensors, dim=None):
        if not isinstance(tensors, list):
            tensors = [tensors]
        for t in tensors:
            if t is None:
                continue
            set_attribute(t, dim, order_type=1)
            if isinstance(t, torch.nn.Parameter):
                # NOTE: Both setattr and __setattr__ doesn't work.
                set_attribute(t.data, dim, order_type=1)

    @classmethod
    def convert_parallel_embedding(cls, module, dim=None):
        """
        Helper function to convert all :class::`torch.nn.Embedding` layers in the model to
        :class::`torch.nn.ParallelEmbedding` layers.
        """
        if not dim in [None, 1, -1, 0]:
            raise ValueError(f"")
        module_output = module

        _ddp_params_and_buffers_to_ignore = []

        if isinstance(module, torch.nn.Embedding):
            module_output = torchshard.nn.ParallelEmbedding(
                module.num_embeddings,
                module.embedding_dim,
                dim=dim
            )
            with torch.no_grad():
                if dim is None:
                    module_output.weight = module.weight
                else:
                    module_output.weight = Parameter(scatter(module.weight, dim=1))

                # Above assignments will brush off the _PARALLEL_DIM. So we set it again.
                cls._set_parameter_attr([module_output.weight], dim)

            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_parallel_embedding(child, dim))

            # make DDP ignore parallel linear's weight and bias parameters
            module_output_child = getattr(module_output, name)
            if isinstance(module_output_child, torchshard.nn.ParallelEmbedding):
                if hasattr(module_output_child.weight, _PARALLEL_DIM):
                    parallel_dim = getattr(module_output_child.weight, _PARALLEL_DIM)
                    if parallel_dim is not None:
                        _ignore_name = f'{name}.weight'
                        _ddp_params_and_buffers_to_ignore.append(_ignore_name)

        del module

        # register for ignoring parallel linear in DDP
        setattr(module_output, '_ddp_params_and_buffers_to_ignore', _ddp_params_and_buffers_to_ignore)

        return module_output
