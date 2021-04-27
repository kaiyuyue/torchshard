import warnings
import math
from typing import Optional, TypeVar, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter

import torchshard as ts

class ParallelArcLoss(Module):
    """
    Parallel ArcFace loss function from :paper:`ArcFace`.
        
    Arguments are similar to :class:`torchshard.nn.ParallelCrossEntropy`. Extra arguments:
    Args:
        None.
        
    Returns:
        loss (Tensor): loss values.
    """

    def __init__(self,
            weight: Optional[Tensor] = None,
            size_average=None,
            ignore_index: int = -100,
            reduce=None,
            reduction: str = 'mean'
        ) -> None:
        super(ParallelArcLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        cos, phi = input
        n, c = cos.shape

        parallel_dim = ts.get_parallel_dim(cos)
        if parallel_dim is not None:
            assert parallel_dim == -1 or parallel_dim == 1

            one_hot = torch.zeros(
                (n, c * ts.distributed.get_world_size()),
                device=cos.device,
                requires_grad=False
            )
            one_hot.scatter_(1, target.unsqueeze(dim=-1), 1)

            # slice one-hot matirx
            one_hot = ts.distributed.scatter(one_hot, dim=-1)
        else:
            one_hot = torch.zeros(
                (n, c),
                device=cos.device,
                requires_grad=False
            )
            one_hot.scatter_(1, target.unsqueeze(dim=-1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cos)

        # set parallel attribute
        ts.register_parallel_dim(output, ts.get_parallel_dim(cos))

        # calculate loss values
        return ts.nn.functional.parallel_cross_entropy(
            output, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )

