#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
mkdir $save_path
hidden=512
nb=2
echo '---start backpack_24_mem_25_split.py   2 batches '
python arxiv_backpack_24_mem_25_split.py \
    --dataset ogbn-arxiv \
    --selection-method arxiv_25_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 18.1 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_${nb}_hidden_${hidden}.log

