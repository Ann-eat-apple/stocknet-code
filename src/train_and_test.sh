#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
rm -rf baselines/stocknet/checkpoints
rm -rf baselines/stocknet/graphs
mkdir baselines/stocknet/checkpoints
mkdir baselines/stocknet/graphs
python2 baselines/stocknet/src/Main.py
