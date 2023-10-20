#!/bin/bash

# mkdir ./log1
save_path=./betty_log/256
mkdir $save_path
hidden=256
nb=2
method=REG
echo "---start Betty_arxiv_e2e_10.py   $nb batches "
python Betty_arxiv_e2e_10.py  \
    --dataset ogbn-arxiv \
    --selection-method $method \
    --num-batch $nb \
    --num-layers 1 \
    --fan-out 10 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/train_loss_nb_${nb}_e2e_h_${hidden}.log
