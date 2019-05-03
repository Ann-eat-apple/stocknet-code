#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
rm -rf ../checkpoints
rm -rf ../graphs
mkdir ../checkpoints
mkdir ../graphs
python2 Main.py
