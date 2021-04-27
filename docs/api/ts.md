 # TORCHSHARD

## Functions

`torchshard`
- [slice](#slice)
- [collect_state_dict](#collect_state_dict)
- [relocate_state_dict](#relocate_state_dict)
- [register_ddp_parameters_to_ignore](#register_ddp_parameters_to_ignore)
- [get_parallel_dim](#get_parallel_dim)
- [register_parallel_dim](#register_parallel_dim)
- [register_parallel_attribute](#register_parallel_attribute)

<p></br></p>

## Tensor Operations

Provides utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.

#### .slice

```python
FUNCTION - torchshard.slice(tensor, num_partitions, dim=None, contiguous=False)
```

Slice a tensor along the specific dimension.

- Parameters
    - **tensor** (Tensor): input tensor.
    - **num_partitions** (int): number of shards to generate.
    - **dim** (int): which dimension along to split (default is `None`).
    - **contiguous** (bool): make output chunks be contiguous (default is `False`).

- Return
    - [tuple](https://docs.python.org/3/c-api/tuple.html). Same as [torch.split](https://pytorch.org/docs/stable/generated/torch.split.html?highlight=split#torch.split) return.

- Examples:

    ```python
    >>> input = torch.empty(3, 10)
    >>> ts.slice(input, 2, dim=-1)
    >>> (tensor1, tensor2) # both of them have the torch.Size([3, 5])
    ```

<p></br></p>

## Serialization

Serialization functions help to easily save and load `model.state_dict()` of parallel layers.

#### .collect_state_dict

```python
FUNCTION - torchshard.collect_state_dict(model, state_dict, prefix='')
```

Returns a dictionary containing a whole state of the module.

Collects tensor shards from multiple GPUs for the parallel layers. Calling this function before `torch.save`.

Both parameters and persistent buffers (e.g. running averages) are included. Keys are corresponding parameter and buffer names.

- Parameters
    - **model** ([nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - Module that contains zero or multiple parallel layers.
    - **state_dict** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)) - produced by `torch.nn.Module.state_dict()`. 
    - **prefix** (Python String) - prefix string you want to add before names of `state_dict()` (default is '').

- Return
    - An ordered dictionary containing a whole state of the module.
    - It will collect parameter shards from all process parallel groups into a unified tensor for the parallel layers.

- Return Type
    - [dict](https://docs.python.org/3/library/stdtypes.html#dict). Same as `model.state_dict()` return.

- Usage
    - If your model has `torchshard.nn` parallel layers, use this `torchshard.collect_state_dict(model)` instead of `model.state_dict()` for saving model states.

- Examples
    
    - for the case with parallel_dim = `-1` or `1`
    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30, dim=-1)
    >>> # m.weight size: [20, 15] on Rank 0
    >>> # m.weight size: [20, 15] on Rank 1
    >>> state_dict = m.state_dict()
    >>> state_dict = ts.collect_state_dict(m, state_dict)
    >>> # In state_dict, m.weight size: [20, 30] on Rank 0 & 1.
    >>> torch.save({'state_dict': state_dict}, 'm.pt')
    ```
    
    - for the case with parallel_dim = `0`
    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30, dim=0)
    >>> # m.weight size: [10, 30] on Rank 0
    >>> # m.weight size: [10, 30] on Rank 1
    >>> state_dict = m.state_dict()
    >>> state_dict = ts.collect_state_dict(m, state_dict)
    >>> # In state_dict, m.weight size: [20, 30] on Rank 0 & 1.
    >>> torch.save({'state_dict': state_dict}, 'm.pt')
    ```
    
    - for the case with parallel_dim = `None`
    ```python
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30, dim=None)
    >>> # m.weight size: [20, 30] on Rank 0 & 1.
    >>> state_dict1 = ts.collect_state_dict(m, m.state_dict(keep_vars=True))
    >>> state_dict2 = m.state_dict(keep_vars=True)
    >>> # state_dict1 and state_dict2 have same keys and values
    >>> torch.save({'state_dict': state_dict}, 'm.pt')
    ```

<p><br/></p>

#### .relocate_state_dict

```python
FUNCTION - torchshard.relocate_state_dict(model, state_dict, prefix='')
```

Loads an object saved with `torch.save()` from a file.

Relocates the state tensors into different GPUs for the parallel layers. Calling this function before `torch.load`.

- Parameters
    - **model** ([nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - Module that contains zero or multiple parallel layers.
    - **state_dict** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)) - produced by `torch.nn.Module.state_dict()`. 
    - **prefix** (Python String) - prefix string you want to add before names of `state_dict()` (default is '').

- Return
    - An ordered dictionary containing a whole state of the module.
    - If the parameter value from state_dict is loaded for a parallel layer, this function will split the value tensor into shards. Return dict only contains the specific shard for the current process parallel group.

- Return Type
    - [dict](https://docs.python.org/3/library/stdtypes.html#dict). Same as `model.state_dict()` return.

- Usage
    - Calling this function before `model.load_state_dict()` and after `torch.load('m.pt')['state_dict']`.
    
- Examples

    - for the case with parallel_dim = `-1` or `1`
    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30, dim=-1)
    >>> state_dict = torch.load('m.pt')['state_dict']
    >>> # In state_dict, m.weight size: [20, 30] on Rank 0 & 1.
    >>> state_dict = ts.relocate_state_dict(m, state_dict)
    >>> # Now in state_dict.
    >>> # m.weight size: [20, 15] on Rank 0
    >>> # m.weight size: [20, 15] on Rank 1
    >>> m.load_state_dict(state_dict)
    ```
    
    - for the case with parallel_dim = `0`
    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30, dim=0)
    >>> state_dict = torch.load('m.pt')['state_dict']
    >>> # In state_dict, m.weight size: [20, 30] on Rank 0 & 1.
    >>> state_dict = ts.relocate_state_dict(m, state_dict)
    >>> # Now in state_dict.
    >>> # m.weight size: [10, 30] on Rank 0
    >>> # m.weight size: [10, 30] on Rank 1
    >>> m.load_state_dict(state_dict)
    ```
    
    - for the case with parallel_dim = `None`
    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30)
    >>> state_dict = torch.load('m.pt')['state_dict']
    >>> # In state_dict, m.weight size: [20, 30] on Rank 0 & 1.
    >>> state_dict = ts.relocate_state_dict(m, state_dict)
    >>> # Now state_dict will keep same as before. So it's OK to ignore calling it.
    >>> m.load_state_dict(state_dict)
    ```

<p><br/></p>

## Utilities

Functions for building model graphs more easily.

#### .register_ddp_parameters_to_ignore

```python
FUNCTION - torchshard.register_ddp_parameters_to_ignore(module)
```

Helper function to register the TorchShard layers into DDP ignoring parameters list.
This function only works with PyTorch >= 1.8.0.

- Parameters
    - **module** (torch.nn.Module): TorchShard will loop all the parameters of the input module and register parallel parameters.

- Return
    - [list](https://docs.python.org/3/tutorial/datastructures.html). It contains parallel parameter names, which will be ignored in DDP.
    
- Examples:

    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume the processes have been initialized.
    >>> # There are 2 model parallel groups and 2 ranks.
    >>> m = torch.nn.ModuleDict({
            'layer1': torch.nn.Linear(20, 20),
            'layer2': ts.nn.ParallelLinear(20, 30)
        })
    >>> check_list = ts.register_ddp_parameters_to_ignore(m)
    >>> print(check_list)
    >>> # check_list = ['layer2.weight', 'layer2.bias']
    >>> m = torch.nn.parallel.DistributedDataParallel(m.cuda())
    ```

<p><br/></p>

#### .get_parallel_dim

```python
FUNCTION - torchshard.get_parallel_dim(tensor)
```

Helper function to get the TorchShard parallel special attribute `torchshard._PARALLEL_ATTR` of tensors.

- Parameters
    - **tensor** (Tensor): Input tensor.
    
- Return
    - Optional [int]: 1, -1, 0, and None.
    
- Examples

    ```python
    >>> input = torch.empty((20, 30))
    >>> m = ts.nn.ParallelLinear(30, 30, dim=-1)
    >>> output = m(input)
    >>> ts.get_parallel_dim(output)
    >>> -1
    >>> ts.get_parallel_dim(input)
    >>> None
    ```

<p><br/></p>

#### .register_parallel_dim

```python
FUNCTION - torchshard.register_parallel_dim(tensor, parallel_dim)
```

Helper function to register the TorchShard parallel special attribute `torchshard._PARALLEL_ATTR` to tensors.

- Parameters
    - **tensor** (Tensor): Input tensor.
    - **parallel_dim** (Optional[int]): Tensor's parallel dimension. It is the value what the `torchshard._PARALLEL_ATTR` will be of the input tensor.
    
- Return
    - Inplace operation.
    
- Notes
    - This function is equivalent to `setattr(tensor, parallel_dim, ts._PARALLEL_ATTR)`.
    - TorchShard functions detect the `torchshard._PARALLEL_ATTR` and then make a right way to calculate forward and backward flows.

- Examples

    ```python
    >>> data = torch.empty((20, 30))
    >>> ts.get_parallel_dim(data)
    >>> None
    >>> ts.register_parallel_dim(data, -1) # indicates it's parallelized in column dim
    >>> ts.get_parallel_dim(data)
    >>> -1
    ```
 
 <p><br/></p>

#### .register_parallel_attribute

```python
FUNCTION - torchshard.register_parallel_attribute(tensor, parallel_dim, order_type)
```

Helper function to register the TorchShard parallel special attributes to tensors.
Attributes have `ts._PARALLEL_DIM`, `ts._ORDER_TYPE`.

- Parameters
    - **tensor** (Tensor): Input tensor.
    - **parallel_dim** (Optional[int]): Tensor's parallel dimension. It is the value what the `torchshard._PARALLEL_ATTR` will be of the input tensor.
    - **order_type** (int): Order type of parameters in parallel layers (default is `0`).
    
    TorchShard has two order types:
    * `0`: input params order [`in_feat_num`, `out_feat_num`] = reverse(`self.weight.shape`)
    * `1`: input params order [`in_feat_num`, `out_feat_num`] = `self.weight.shape`

    
- Return
    - Inplace operation.
    
- Notes
    - This fine-grained op is only for writing a customized layer.

- Examples

    ```python
    >>> param = torch.empty((20, 30))
    >>> ts.register_parallel_attribute(param, -1, 1)
    >>> ts.get_parallel_dim(data)
    >>> -1
    ```

<p><br/></p>

<p>&#10141; Back to the <a href="../">main page</a></p>
