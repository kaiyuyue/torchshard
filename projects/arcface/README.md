# Large-Scale Parallel ArcFace Training

[[`tutorial`](../../docs/tutorial/face.md)]

## Datasets

| datasets | links | file type |
| :---: | :---: | :--: |
| MS1M | [ [Baidu Pan](https://pan.baidu.com/s/1oVZQ3SQR8x3CMqcAAkGLMg) ] - `vkco` | image |
| LFW | [ [Baidu Pan](https://pan.baidu.com/s/1thV_cY8s96YPuUIyEDNtxw) ] - `35b2` | image |

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

## Validation

To test the model, run:

```bash
python3 valid.py --batch-size 1024 -workers 16 --imgs-root data/lfw/imgs --pair-list data/lfw/pair.list --resume checkpoint.pth.tar
```
