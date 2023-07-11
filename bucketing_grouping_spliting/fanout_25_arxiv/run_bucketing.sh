#!/bin/bash

# mkdir ./log1
save_path=../bucketing_log
mkdir $save_path
echo '---start backpack_24_mem_25_split.py   4 batches '
python arxiv_backpack_24_mem_25_split.py \
    --dataset ogbn-arxiv \
    --selection-method arxiv_25_backpack_bucketing \
    --num-batch 4 \
    --mem-constraint 18 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 1024 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_4_bucketing_1024.log

# echo '---start backpack_24_mem_25_split_time.py   15 batches '
# python backpack_24_mem_25_split_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_bucketing \
#     --num-batch 15 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_15_bucketing.log

# echo '---start backpack_24_mem_25_split_time.py REG  13 batches '
# python backpack_24_mem_25_split_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_bucketing \
#     --num-batch 13 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_13_bucketing.log

# echo '---start backpack_24_mem_25_split_time.py REG  12 batches '
# python backpack_24_mem_25_split_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_bucketing \
#     --num-batch 12 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_12_bucketing.log
