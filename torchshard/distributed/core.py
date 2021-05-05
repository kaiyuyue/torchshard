import torch

from ..utils import check_divisibility

_PARALLEL_GROUP = None
_PARALLEL_SIZE = None

def is_initialized():
    return _PARALLEL_GROUP is not None


def get_group():
    assert _PARALLEL_GROUP is not None, 'model parallel process group is not initialized.'
    return _PARALLEL_GROUP


def get_world_size():
    return torch.distributed.get_world_size()


def get_rank():
    return torch.distributed.get_rank(group=get_group())


def get_group_size():
    assert _PARALLEL_GROUP is not None, 'model parallel process group is not initialized.'
    return _PARALLEL_SIZE


def init_process_group(group_size: int = -1) -> None:
    assert torch.distributed.is_initialized(), \
        "Default process group has not been initialized, " \
        "please make sure to call init_process_group."
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    assert group_size >= 1, \
        "group_size must in [1, world_size], but got {}".format(group_size)
    global _PARALLEL_SIZE
    _PARALLEL_SIZE = min(group_size, world_size)
    check_divisibility(world_size, _PARALLEL_SIZE)


    global _PARALLEL_GROUP
    assert _PARALLEL_GROUP is None, 'model parallel process group is already initialized.'
    for i in range(world_size // _PARALLEL_SIZE):
        ranks = range(i * _PARALLEL_SIZE, (i+1) * _PARALLEL_SIZE)
        group = torch.distributed.new_group(ranks)
        if i == rank // _PARALLEL_SIZE:
            _PARALLEL_GROUP = group


def destroy_process_group():
    global _PARALLEL_GROUP
    _PARALLEL_GROUP = None

    global _PARALLEL_SIZE
    _PARALLEL_SIZE = None
