# Scaling up minGPT Training

[[`tutorial`](../../docs/tutorial/mgpt.md)] [[`official repo`](https://github.com/karpathy/minGPT)]

## Training

Experiments here keep the same default configurations as minGPT to reproduce its results.
GPT parameters are set in `--n-layer 12 --n-head 8 --n-embd 256`.

To control how many GPUs and which part of them to use, do:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

To train GPT with DP mode, run:

```bash
python3 main.py --batch-size 64 ${dataset directory} 
```

To train GPT with DDP mode, run:

```bash
python3 main.py --multiprocessing-distributed --batch-size 72 ${dataset directory}
```

To train GPT with **TorchShard**, run:

```bash
python3 main.py --multiprocessing-distributed --batch-size 400 --enable-model-parallel --enable-one-data-group ${dataset directory} 
```

To train GPT with **TorchShard + AMP**, run:

```bash
python3 main.py --multiprocessing-distributed --batch-size 400 --enable-model-parallel --enable-one-data-group --enable-amp-mode ${dataset directory} 
```

To train GPT with **TorchShard + ZeRO**, run:

```bash
python3 main.py --multiprocessing-distributed --batch-size 400 --enable-model-parallel --enable-one-data-group --enable-zero-optim ${dataset directory} 
```

**Note**: ZeRO optimizer comes with PyTorch >= 1.9.0.

## Validation

Put [paly_image.ipynb](https://github.com/karpathy/minGPT/blob/master/play_image.ipynb) in this folder.
Skip the training step and run the inference part to check out visualization results.

## Benchmark

Here are some scripts to run benchmarks for comparisons.

To run with DDP:

```bash
python3 main.py --multiprocessing-distributed --batch-size 16 ${dataset directory} 
```

To run with TorchShard:

```bash
python3 main.py --multiprocessing-distributed --batch-size 16 --enable-model-parallel --enable-one-data-group ${dataset directory} 
```

To run with AMP:

```bash
python3 main.py --multiprocessing-distributed --batch-size 16 --enable-amp-mode ${dataset directory}
```

To run with ZeRO:

```bash
python3 main.py --multiprocessing-distributed --batch-size 16 --enable-zero-optim ${dataset directory}
```

To run with TorchShard + AMP or ZeRO:

add `--enable-amp-mode` or `--enable-zero-optim` to the following script:

```bash
python3 main.py --multiprocessing-distributed --batch-size 16 --enable-model-parallel --enable-one-data-group ${dataset directory} 
```
