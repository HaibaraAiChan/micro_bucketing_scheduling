#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
mkdir $save_path
# echo '---start backpack_24_mem_25_split.py   4 batches '
# python arxiv_backpack_24_mem_25_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_25_backpack_bucketing \
#     --num-batch 4 \
#     --mem-constraint 18 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 1024 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_4_bucketing_1024_new.log
# echo '---start backpack_24_mem_25_split.py  5 batches '
# python arxiv_backpack_24_mem_25_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_25_backpack_bucketing \
#     --num-batch 5 \
#     --mem-constraint 18 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 1024 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_5_bucketing_1024_new_.log
echo '---start backpack_24_mem_25_split.py  6 batches '
python arxiv_backpack_24_mem_25_split.py \
    --dataset ogbn-arxiv \
    --selection-method arxiv_25_backpack_bucketing \
    --num-batch 6 \
    --mem-constraint 18 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 1024 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_6_bucketing_1024___.log