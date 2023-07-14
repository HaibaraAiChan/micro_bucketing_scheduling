#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
# mkdir $save_path

# echo '---start backpack_29_mem_30_split.py 5 batches '
# python arxiv_backpack_29_mem_30_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_30_backpack_bucketing \
#     --num-batch 5 \
#     --mem-constraint 18 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 1 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_5_bucketing_h_256.log
echo '---start backpack_29_mem_30_split.py 6 batches '
python arxiv_backpack_29_mem_30_split.py \
    --dataset ogbn-arxiv \
    --selection-method arxiv_30_backpack_bucketing \
    --num-batch 6 \
    --mem-constraint 18 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 1 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_6_bucketing_h_256.log