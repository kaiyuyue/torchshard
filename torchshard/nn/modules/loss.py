import math
import logging
from typing import Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from .. import functional as F
from .. import _reduction as _Reduction
from ...distributed import gather

# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


class ParallelCrossEntropyLoss(_WeightedLoss):
    """
    Parallel version of :class::`torch.nn.CrossEntropy`.

    Arguments are similar to :class:`torch.nn.CrossEntropy`. Extra arguments:
    Args:
        None.

    Returns:
        :class::`torch.Tensor`: loss values.
    """
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(ParallelCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return F.parallel_cross_entropy(input, target, weight=self.weight,
                                        ignore_index=self.ignore_index, reduction=self.reduction)

