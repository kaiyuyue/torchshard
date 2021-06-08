# Large-Scale Parallel ArcFace Training

[[`tutorial`](../../docs/tutorial/face.md)]

## Datasets

| datasets | links | file type |
| :---: | :---: | :--: |
| MS1M | [ [Baidu Pan](https://pan.baidu.com/s/1KtvmLk6CS4mJQXtcy_WGjQ) ] - `u7rb` | image |
| LFW | [ [Baidu Pan](https://pan.baidu.com/s/1zl3yMF8iLQeygV0SGNzw3g) ] - `k2jm` | image |

Images have been aligned and resized to 112 x 112.
Put them into `./data` folder, which looks like the following structure:

```
`-- data
  |`-- ms1m
    |-- 15907
    |-- 45600
    |-- ...
  |`-- lfw
    |-- pair.list
    |-- imgs
```

## Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single Node

To control how many GPUs and which part of them to use, do:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

To train a base model, run:

```bash
python3 main.py --multiprocessing-distributed --num-classes 85744 ${ms1m directory} 
```

To train a model with **TorchShard**, run:

```bash
python3 main.py --multiprocessing-distributed --num-classes 85744 --enable-model-parallel --model-parallel-dim 1 ${ms1m directory} 
```

To train a model with **TorchShard + AMP**, run:

```bash
python3 main.py --multiprocessing-distributed --num-classes 85744 --enable-model-parallel --model-parallel-dim 1 --enable-amp-mode ${ms1m directory} 
```

To train a model with **TorchShard + ZeRO**, run:

```bash
python3 main.py --multiprocessing-distributed --num-classes 85744 --enable-model-parallel --model-parallel-dim 1 --enable-zero-optim ${ms1m directory} 
```

**Note**: ZeRO optimizer comes with PyTorch >= 1.9.0.

### Multiple Nodes

Each node has 8 GPUs. We train a **I**ResNet-50 whose fully-connected layer size is 4000000 (4 million). To train a model with 2 nodes:

- Node 0

```bash
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=ens1f1  # SOCKET_OF_NODE0
export MASTER_ADDR=172.20.10.22   # IP_OF_NODE0
export MASTER_PORT=12355          # FREEPORT

python3 main.py \
    ${ms1m directory} \
    --multiprocessing-distributed --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
    --enable-model-parallel --model-parallel-dim 1 \
    --batch-size 128 --num-classes 4000000 \
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
    ${ms1m directory} \
    --multiprocessing-distributed --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
    --enable-model-parallel --model-parallel-dim 1 \
    --batch-size 128 --num-classes 4000000 \
    --world-size 2 --rank 1 \
    # --enable-amp-mode or --enable-zero-optim 
```

## Validation

To test the model, run:

```bash
python3 valid.py --batch-size 1024 -workers 16 --imgs-root data/lfw/imgs --pair-list data/lfw/pair.list --resume checkpoint.pth.tar
```
