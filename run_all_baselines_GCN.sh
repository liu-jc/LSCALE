#!/usr/bin/env bash
for j in random uncertainty largest_degrees featprop
do
  for i in cora citeseer pubmed
  do
    echo "run $j in $i"
    CUDA_VISIBLE_DEVICES=1 python run_baselines.py --dataset $i --model GCN --epoch 300 --strategy $j --file_io 1 --lr 0.01 --hidden 16
  done
done

