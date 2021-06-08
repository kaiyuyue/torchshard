# Large-Scale Parallel Training on ImageNet

[[`tutorial`](../../docs/tutorial/in1k.md)] [[`official repo`](https://github.com/pytorch/examples/tree/master/imagenet)]

## Training and Validation

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single Node

To control how many GPUs and which part of them to use, do:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

To train a base model, run:

```bash
python3 main.py --multiprocessing-distributed --num-classes 1000 ${imagenet directory} 
```

To train a model with **TorchShard**, run:

```bash
python3 main.py --multiprocessing-distributed --num-classes 1000 --enable-model-parallel --model-parallel-dim 0 ${imagenet directory} 
```

**Note**: `--model-parallel-dim 0` can also be set in `-1` or `1`.

To train a model with **TorchShard + AMP**, run:

```bash
python3 main.py --multiprocessing-distributed --num-classes 1000 --enable-model-parallel --model-parallel-dim 0 --enable-amp-mode ${imagenet directory} 
```

To train a model with **TorchShard + ZeRO**, run:

```bash
python3 main.py --multiprocessing-distributed --num-classes 1000 --enable-model-parallel --model-parallel-dim 0 --enable-zero-optim ${imagenet directory}
```

**Note**: ZeRO optimizer comes with PyTorch >= 1.9.0.

### Multiple Nodes

Each node has 8 GPUs. We train a ResNet-50 whose fully-connected layer size is 2000000 (2 million). To train a model with 2 nodes:

- Node 0

```bash
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=ens1f1  # SOCKET_OF_NODE0
export MASTER_ADDR=172.20.10.22   # IP_OF_NODE0
export MASTER_PORT=12355          # FREEPORT

python3 main.py \
    ${imagenet directory} \
    --multiprocessing-distributed --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
    --enable-model-parallel --model-parallel-dim 1 \
    --batch-size 128 --num-classes 2000000 \
    --world-size 2 --rank 0 \
    # --enable-amp-mode or --enable-zero-optim 
```

- Node 1

```bash
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=ens1f1  # SOCKET_OF_NODE0
export MASTER_ADDR=172.20.10.22   # IP_OF_NODE0
export MASTER_PORT=12355          # FREEPORT

python3 main.py \
    ${imagenet directory} \
    --multiprocessing-distributed --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
    --enable-model-parallel --model-parallel-dim 1 \
    --batch-size 128 --num-classes 2000000 \
    --world-size 2 --rank 1 \
    # --enable-amp-mode or --enable-zero-optim 
```

## Scaling-up Training

Super large-scale training with TorchShard enjoys massive GPUs.
The training can be scaled up by

- increasing classes number like from 1000 to 1000000 (1 million).
- bulding larger networks, for example, using the [Transformer](https://arxiv.org/abs/1706.03762) blocks.