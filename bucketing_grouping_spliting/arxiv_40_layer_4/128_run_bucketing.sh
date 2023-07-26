#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log/128
# mkdir $save_path
nb=4
echo "---start backpack.py ${nb} batches "
python arxiv_backpack.py \
    --dataset ogbn-arxiv \
    --selection-method arxiv_40_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 18.2 \
    --num-layers 4 \
    --fan-out 10,25,30,40 \
    --num-hidden 128 \
    --num-runs 1 \
    --num-epoch 1 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_${nb}_bucketing_h_128.log
