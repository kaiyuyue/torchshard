#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python3 -m unittest discover -v -s tests