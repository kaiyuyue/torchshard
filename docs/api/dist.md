# TORCHSHARD.DISTRIBUTED

## Functions

`torchshard.distributed`
- [is_initialized ( )](#is_initialized) 
- [init_process_group ( )](#init_process_group)
- [destroy_process_group ( )](#destroy_process_group)
- [scatter ( )](#scatter)
- [gather ( )](#gather)
- [reduce ( )](#reduce)
- [copy ( )](#copy) 
- [get_rank ( )](#get_rank)
- [get_world_size ( )](#get_world_size)
- [get_group ( )](#get_group)
- [get_group_size ( )](#get_group_size)

## Backends

`torchshard.distributed` package only works with the PyTorch built-in backend `NCCL` for distributed GPU training.

## Initialization

The package needs to be initialized using the `torch.distributed.init_process_group()` function and `torchshard.distributed.init_process_group()` before calling any other methods. This blocks until all processes have joined.
The model parallel group processes from `torchshard.distributed` package will keep the same `world_size` and `rank` with processes from `torch.distributed` package.

#### .is_initialized

```python
FUNCTION - torchshard.distributed.is_initialized()
```

Helps to check if the default process group has been initialized.

<p><br/></p>

#### .init_process_group

```python
FUNCTION - torchshard.distributed.init_process_group(group_size=-1)
```

Initializes the default distributed process parallel group.

- Parameters
    - **group_size** (int) - Number of model parallel groups participating in the job. 

- Returns
    - None.
    
- Examples

    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume we would like to have 2 model parallel groups and 2 ranks.
    >>> torch.distributed.init_process_group(
            backend='nccl', 
            init_method='tcp://127.0.0.1:8000',
            world_size=2,
            rank=0 # on Rank 0 / rank=1 on Rank 1 
        )
    >>> ts.distributed.init_process_group(group_size=2)
    ```

<p><br/></p>

#### .destroy_process_group

```python
FUNCTION - torchshard.distributed.destroy_process_group()
```

Destropy all the process model parallel groups.

<p><br/></p>

#### .get_group

```python
FUNCTION - torchshard.distributed.get_group()
```

Returns the model parallel group the caller rank belongs to. Created by `torch.distributed.new_group`.

- Parameters
    - None.

- Returns
    - A handle of distributed group that can be given to collective calls.
    
<p><br/></p>

#### .get_rank

```python
FUNCTION - torchshard.distributed.get_rank()
```

Returns the rank of current process model parallel group.

- Parameters
    - None.

- Returns
    - The rank of the process group -1, if not part of the group.

- Examples

    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> current_rank = ts.distributed.get_rank()
    >>> current_rank
    >>> 0 # Rank 0
    >>> 1 # Rank 1 
    ```

<p><br/></p>

#### .get_world_size

```python
FUNCTION - torchshard.distributed.get_world_size()
```

Returns the number of processes in the current process model parallel group.

- Parameters
    - None.

- Returns
    - The world size of the process group -1, if not part of the group.

- Examples

    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> world_size = ts.distributed.get_world_size()
    >>> world_size
    >>> 2 # Rank 0
    >>> 2 # Rank 1 
    ```

<p><br/></p>

#### .get_group_size

```python
FUNCTION - torchshard.distributed.get_group_size()
```

Returns the group size in the current process model parallel group.

- Parameters
    - None.

- Returns
    - The group size of the process group.

- Examples

    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> group_size = ts.distributed.get_group_size()
    >>> group_size
    >>> 2 # Rank 0
    >>> 2 # Rank 1 
    ```

<p><br/></p>

## Multi-GPU Collective Functions

- Every following collective operation function is in the default `async_op` mode - False.
- Different with `torch.distributed` collective op functions, every following collective operation function only matters two parameters - input `tensor` and parallel `dim`. The input tensor is a torch.Tensor not list. The parallel dim could be one of `None`, column `1`, column `-1`, and row `0`.

#### .scatter

```python
FUNCTION - torchshard.distributed.scatter(tensor, dim=-1)
```

Scatters a tensor to all processes in a model parallel group. Each process will receive one shard and store its data in the output tensor.

- Parameters
    - **tensor** (Tensor) - Input tensor.
    - **dim** (int) - Dimension along which to scatter the tensor (default is last dim `-1`).

- Returns
    - Async work handle because this is an async op.

- Examples

    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> input = torch.Tensor(6, 8)
    >>> input.shape
    torch.Size([6, 8])
    >>> output = ts.distributed.scatter(input, dim=0)
    >>> output.shape 
    torch.Size([3, 8])
    >>> output = ts.distributed.scatter(input, dim=1)
    >>> output.shape 
    torch.Size([6, 4])
    ```
    
<p><br/></p>

#### .gather

```python
FUNCTION - torchshard.distributed.gather(tensor, dim=-1)
```

Gathers parallel tensors in a single process.

- Parameters
    - **tensor** (Tensor) - Input tensor.
    - **dim** (int) - Dimension along which to gather the tensor (default is last dim `-1`).

- Returns
    - Async work handle because this is an async op.

- Examples

    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> input = torch.Tensor(6, 8)
    >>> input.shape
    torch.Size([6, 8])
    >>> output = ts.distributed.gather(input, dim=0)
    >>> output.shape 
    torch.Size([12, 8])
    >>> output = ts.distributed.gather(input, dim=1)
    >>> output.shape 
    torch.Size([6, 16])
    ```

<p><br/></p>

#### .reduce

```python
FUNCTION - torchshard.distributed.reduce(tensor)
```

Sum reduces the tensor data across all machines in such a way that all get the final result. Same as `torch.distributed.all_reduce` with `op=torch.distributed.ReduceOp.SUM`. Complex tensors are supported.

- Parameters
    - **tensor** (Tensor) - Input and output of the collective. The function operates in-place.

- Returns
    - Async work handle because this is an async op.

- Examples

    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    >>> tensor
    tensor([1, 2]) # Rank 0
    tensor([3, 4]) # Rank 1
    >>> ts.distributed.reduce(tensor)
    >>> tensor
    tensor([4, 6]) # Rank 0
    tensor([4, 6]) # Rank 1
    ```
    
<p><br/></p>

#### .copy

```python
FUNCTION - torchshard.distributed.copy(tensor)
```

Copys a tensor to reduce its gradients across all machines during backward.

- Parameters
    - **tensor** (Tensor) - Input tensor.

- Returns
    - Async work handle because this is an async op.

- Examples

    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> input = torch.Tensor(6, 8)
    >>> input = ts.distributed.copy(input)
    >>> # In forward, input will be kept same as before on each rank.
    >>> # In backward, input gradients will be sum reduced across all the ranks.
    ```

<p><br/></p>

<p>&#10141; Back to the <a href="../">main page</a></p>
