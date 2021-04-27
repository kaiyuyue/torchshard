# Installation

## Requirements

- Linux or macOS with Python ≥ 3.6.
- PyTorch ≥ 1.8.0 is recommended.

## Build TorchShard from Source

```bash
python -m pip install 'git+https://github.com/KaiyuYue/torchshard.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/KaiyuYue/torchshard.git
python -m pip install -e torchshard

# clean
rm -fr ./build ./torchshard.egg-info
```

## Install from PyPi

```bash
pip install torchshard
```
