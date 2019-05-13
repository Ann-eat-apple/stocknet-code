#!/bin/sh
export CUDA_VISIBLE_DEVICES=0 
rm -rf baselines/stocknet/checkpoints 
rm -rf baselines/stocknet/graphs 
mkdir baselines/stocknet/checkpoints 
mkdir baselines/stocknet/graphs 
python2 baselines/stocknet/src/Main.py | tee data/StockNet/logs/$1.k_$2.$3.$4.log