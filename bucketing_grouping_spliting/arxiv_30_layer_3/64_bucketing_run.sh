#!/bin/bash

# mkdir ./log1
hidden=64
save_path=./bucketing_log/${hidden}
# mkdir $save_path
nb=2
echo "---start backpack_29_mem_30_split.py ${nb} batches "
python arxiv_backpack_29_mem_30_split.py \
    --dataset ogbn-arxiv \
    --selection-method arxiv_30_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 18.2 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_${nb}_bucketing_h_${hidden}.log
