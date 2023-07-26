#!/bin/bash

# mkdir ./log1
save_path=./betty_log/128
mkdir $save_path
echo '---start Betty_arxiv_e2e_30.py   1 batches '
python Betty_arxiv_e2e_30.py  \
    --dataset ogbn-arxiv \
    --selection-method REG \
    --num-batch 1 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 128 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_1_e2e_128.log
