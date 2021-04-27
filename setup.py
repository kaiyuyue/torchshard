from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

import torchshard as ts
version_suffix = '' # e.g. '.dev', '.pre-release'

setup(
    name="torchshard",
    version=str(ts.__version__) + version_suffix,
    description='Torchshard: Slicing a PyTorch Tensor Into Parallel Shards.',
    author="Kaiyu Yue",
    author_email="kaiyuyue@umd.edu",
    long_description = "TorchShard is a lightweight engine for slicing a PyTorch tensor into parallel shards. It can reduce GPU memory and scale up the training when the model has massive linear layers (e.g., ViT, BERT and GPT) or huge classes (millions). It has the same API design as PyTorch.",
    long_description_content_type="text/x-rst",
    url="https://github.com/KaiyuYue/torchshard",
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
