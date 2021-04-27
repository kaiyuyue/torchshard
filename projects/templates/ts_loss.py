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


class ParallelLoss(Module):
    """
    Build a customized parallel loss function.

    Steps:
        0. change the loss function's name
        1. add / delete additional input parameters for it
        2. write the forward flow.
    """

    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self,
            weight: Optional[Tensor] = None,
            size_average=None,
            ignore_index: int = -100,
            reduce=None,
            reduction: str = 'mean'
        ) -> None:
        super(ParallelLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)

        # the forward code ... _forward
        input = _forward(input)

        # pass final result to torchshard.nn.functional ops to calculate loss values
        return ts.nn.functional.parallel_cross_entropy(input, target, weight=self.weight,
                                        ignore_index=self.ignore_index, reduction=self.reduction)

