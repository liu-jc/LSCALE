#!/usr/bin/env bash
for i in cora citeseer pubmed CS Physics
do
    echo "run LSCALE in $i"
    CUDA_VISIBLE_DEVICES=0 python LSCALE.py --dataset $i --epoch 300 --strategy LSCALE --file_io 1 --reweight 0 --hidden 100 --feature cat --adaptive 1 --weight_decay 0.000005
done
