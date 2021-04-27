# TORCHSHARD.NN

## Classes and Functions

`torchshard.nn`
- [ParallelLinear](#nnparallellinear)
    - [convert_parallel_linear ( )](#convert_parallel_linear)
- [ParallelCrossEntropyLoss](#nnparallelcrossentropyloss)
- [RegisterParallelDim](#nnregisterparalleldim)
- [ParallelEmbedding](#nnembedding)

`torchshard.nn.functional`
- [functional.parallellinear ( )](#parallel_linear)
- [functional.parallel_cross_entropy ( )](#parallel_cross_entropy_loss)
- [functional.parallel_embedding ( )](#parallel_embedding)

`torchshard.nn.init`
- [shard_init_helper_](#shard_init_helper_)

<p><br/></p>

## Linear Layers

Parallel linear layers has two capabilities: parallel way and non-parallel way. In non-parallel way, they behavior as same as PyTorch original linear layers.

#### nn.ParallelLinear

```python
CLASS - torchshard.nn.ParallelLinear(in_features, out_features, bias=True, dim=None)
```

Parallel version of [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear).
Applies a linear transformation to the incoming tensor: `y = xA^T + b` in the parallel manner.

This module supports [TensorFloat32](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere).

- Parameters
    - **in_features** (int) - size of each input sample.
    - **out_features** (int) - size of each output sample.
    - **bias** (bool) - If set to `False`, the layer will not learn an additive bias. Default: `True`.
    - **dim** (int) - Dimension along which to split weight and bias into shards (default is `None`).
    
        | dim | behavior | 
        | --- | --- | 
        | `None`    | `nn.ParallelLinear` = `nn.Linear` | 
        | `1` or `-1` | `nn.ParallelLinear` will be parallel in **column** |
        | `0` | `nn.ParallelLinear` will be parallel in **row** |

- Shape
    - If `dim` = `None`, both input and output are same as `torch.nn.Linear`. Input: `(N, *, H_in)` where N is batch size number, * means any number of additional dimensions, and `H_in = in_features`. Output: `(N, *, H_out)` where `H_out = out_features`.
    - If `dim` = `-1` or `1`, input shape is `(N, *, H_in)` and output shape is `(N, *, int(H_out / group_size))`. `group_size` is fixed after calling `torchshard.distributed.init_process_group()`.
    - If `dim` = `0`, input shape is `(N, *, int(H_in / group_size))` and output shape is `(N, *, H_out)`.
    
- Variables
    - **~ParallelLinear.weight** - the learnable weights of the module of shape (out_features, in_features). The values are initialized as same as the values of **~torch.nn.Linear.weight**.
    - **~ParallelLinear.bias** - the learnable bias of the module of shape (out_features). If `bias` is `True`, the values are initialized as same as the values of **~torch.nn.Linear.bias**.

- Examples
    
    - for the case with parallel `dim` = `None`
    
    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume we would like to have 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30, bias=True, dim=None)
    >>> input = torch.randn(128, 20)
    >>> output = m(input)
    >>> print(output.size)
    torch.Size([128, 30])
    ```
    
    - for the case with parallel `dim` = `0`
    
    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume we would like to have 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30, bias=True, dim=0)
    >>> input = torch.randn(128, 20)
    >>> output = m(input)
    >>> print(output.size)
    torch.Size([128, 30])
    ```
    
     - for the case with parallel `dim` = `1` / `-1`
    
    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume we would like to have 2 model parallel groups and 2 ranks.
    >>> m = ts.nn.ParallelLinear(20, 30, bias=True, dim=-1)
    >>> input = torch.randn(128, 20)
    >>> output = m(input)
    >>> print(output.size)
    torch.Size([128, 15])
    ```

<p><br/></p>
 
#### .convert_parallel_linear

```python
CLASSMETHOD - torchshard.nn.ParallelLinear.convert_parallel_linear(module, dim=None)
```

Helper function to convert all `torch.nn.Linear` layers in the module to `torchshard.nn.ParallelLinear` layers with `dim`.

- Parameters
    - **module** ([nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - module containing one or more `torch.nn.Linear` layers.
    - **dim** (int) - Dimension along which to split weight and bias into shards (default is `None`). 
    
- Returns
    - The original module with the converted `torchshard.nn.ParallelLinear` layers. If the original module is just a `torch.nn.Linear` layer, a new `torchshard.nn.ParallelLinear` layer object will be returned instead.
    - This conversion will NOT hurt the original parameters' (weight and bias) values, if they are initialized.

- Eamples

    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> # Assume we have already initialized parallel process group
    >>> m = torch.nn.Linear(20, 30)
    >>> m
    Linear(in_features=20, out_features=30, bias=True)
    >>> m = ts.nn.ParallelLinear.convert_parallel_linear(m, dim=-1)
    >>> m
    ParallelLinear(in_features=20, out_features=30, bias=True, dim=-1)
    ```

<p><br/></p>

## Sparse Layers

Parallel version of [sparse layers](https://pytorch.org/docs/stable/nn.html#sparse-layers).

#### nn.Embedding

```python
CLASS - torchshard.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, dim=None)
```

A simple lookup table that stores embeddings of a fixed dictionary and size.
`dim` controls which dimension along to make weight parallel.

- Parameters
    - Parameters are same as [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding).
    - **dim** (int): which dimension along to make weight parallel (default is `None`).
    It can be set in `None`, `1`, `-1`, and `0`.

- Variables
    **~Embedding.weight** (Tensor) – the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from N(0,1).
    
- Examples:

    ```python
    >>> import torchshard as ts
    >>> m = ts.nn.ParallelEmbedding(64, 512, dim=1) # scatter Embedding weight in column
    >>> m = ts.nn.ParallelEmbedding(64, 512, dim=0) # scatter Embedding weight in row
    >>> m = ts.nn.ParallelEmbedding(64, 512) # = torch.nn.Embedding(64, 512)
    ```

<p><br/></p>

## Loss Layers

Parallel loss layers has two capabilities: parallel way and non-parallel way. In non-parallel way, they behavior as same as PyTorch original loss layers.

#### nn.ParallelCrossEntropyLoss

```python
CLASS - torchshard.nn.ParallelCrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

Parallel version of `torch.nn.CrossEntropyLoss`. 
More details in [torch.nn.CrossEntropy Docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).

- Parameters
    - Totally same as the input parameters of `torch.nn.CrossEntropyLoss`.

- Shape
    - Totally same as the shape of `torch.nn.CrossEntropyLoss`.

- Usage
    - `nn.ParallelCrossEntropyLoss` usually pairs with `nn.ParallelLinear` to use in our training. 
    `nn.ParallelCrossEntropyLoss` can automatically recognize the output tensor shards from `nn.ParallelLinear`.
    If it is used with `torch.nn.Linear`, it will be same as `torch.nn.CrossEntropy`.

- Examples

    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> loss = ts.nn.ParallelCrossEntropyLoss()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.empty(3, dtype=torch.long).random_(5)
    >>> output = loss(input, target)
    >>> output.backward()
    ```

<p><br/></p>

## Utilities

Layers for building model graphs more easily. 

#### nn.RegisterParallelDim

```python
CLASS - torchshard.nn.RegisterParallelDim(dim=None)
```

nn.Module for registering the parallel dimension into tensors.

- Parameters
    - **dim** (int): the dimension which will be registered into tensors (default is None). It can be `0`, `-1`, `1`, and `None`.

- Type
    - `torch.nn.Module`.

- Examples:
       
    ```python
    >>> input = torch.Tensor(3)
    >>> ts.get_parallel_dim(input)
    >>> None
    >>> m = ts.nn.RegisterParallelDim(dim=0)
    >>> input = m(input)
    >>> ts.get_parallel_dim(input)
    >>> 0 
    ```

<p><br/></p>

## Functional Layers

`torchshard.nn.functional` version of `torchshard.nn` class methods.

#### .parallel_linear

```python
FUNCTION - torchshard.nn.functional.parallel_linear(input, weight, bias=None, dim=None)
```

Parallel version of `torch.nn.functional.linear`. This operator supports [TensorFloat32](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere).

- Usage
    - If `dim` = `None`, this layer is same as `torch.nn.functional.linear`.
    - If `dim` is NOT `None`, this layer will perform in a parallel manner accross process parallel groups. It can NOT make its weight and bias into parallel shards by itself. You need prepare weight and bias shards rightly before calling it.
    - **Recommend to use** `torchshard.nn.ParallelLinear` **to build your model instead of this layer.**

<p><br/></p>

#### .parallel_cross_entropy_loss

```python
FUNCTION - torchshard.nn.functional.cross_entropy_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

Parallel version of `torch.nn.functional.cross_entropy`. More details in [torch.nn.functional.cross_entropy Docs](https://pytorch.org/docs/stable/nn.functional.html#cross-entropy).

- Parameters
    - Totally same as the input parameters of `torch.nn.functional.cross_entropy`.
    
- Usage
    - This functional loss layer is same as `torchshard.nn.CrossEntropy`. 

- Examples

    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randint(5, (3,), dtype=torch.int64)
    >>> loss = ts.nn.functional.parallel_cross_entropy(input, target)
    >>> loss.backward()
    ```

#### .parallel_embedding

```python
CLASS - torchshard.nn.functional.parallel_embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, num_embeddings=None, dim=None)
```

A simple lookup table that stores embeddings of a fixed dictionary and size.
`dim` controls which dimension along to make weight parallel.

See [torchshard.nn.ParallelEmbedding](#nnembedding) for more details.

- Parameters
    - Parameters are same as [torchshard.nn.ParallelEmbedding](#nnembedding).
    - **num_embeddings** (int): embedding size (default is `None`).
    - **dim** (int): which dimension along to make weight parallel (default is `None`).
    It can be set in `None`, `1`, `-1`, and `0`.

- Variables
    **~Embedding.weight** (Tensor) – the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from N(0,1).
    
- Examples:

    ```python
    >>> import torchshard as ts
    >>> input = torch.tensor([[1,2,4,5],[4,3,2,9]])
    >>> embedding_matrix = torch.rand(5, 3) # on Rank 0
    >>> embedding_matrix = torch.rand(5, 3) # on Rank 1
    >>> ts.nn.functional.parallel_embedding(input, embedding_matrix, num_embeddings=4, dim=0)
    ```

<p><br/></p>

## Init Layers

Helper functions to initialize tensors of parallel layers.

#### .shard_init_helper_

```python
FUNCTION - torchshard.nn.shard_init_helper_(init_method, tensor, **kwargs)
```

This helper function can be used to initialize tensor shards across all process parallel groups.

- Parameters
    - **init_method** ([torch.nn.init](https://pytorch.org/docs/stable/nn.init.html)) - Methods from `torch.nn.init`.
    - **tensor** (Tensor) - a `torch.Tensor`. Can be 2D weight and 1D bias.
    - ** **kwargs** - Any input parameters for above `init_method` that is called.
    
- Examples

    ```python
    >>> import torch
    >>> import torchshard as ts
    >>> m = ts.nn.ParallelLinear(20, 30, dim=-1)
    >>> # init weight
    >>> ts.nn.init.shard_init_helper_(
            torch.nn.init.kaiming_normal_,
            m.weight,
            a=0, mode='fan_in', nonlinearity='leaky_relu'
        )
    >>> # init bias
    >>> ts.nn.init.shard_init_helper_(
            torch.nn.init.constant_,
            m.bias,
            val=0.1
        )
    ```

<p><br/></p>

<p>&#10141; Back to the <a href="../">main page</a></p>
