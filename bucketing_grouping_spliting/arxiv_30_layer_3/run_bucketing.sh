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
# echo '---start backpack_29_mem_30_split.py 6 batches '
# python arxiv_backpack_29_mem_30_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_30_backpack_bucketing \
#     --num-batch 6 \
#     --mem-constraint 18 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 1 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_6_bucketing_h_256.log
#     echo '---start backpack_29_mem_30_split.py 7 batches '
# python arxiv_backpack_29_mem_30_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_30_backpack_bucketing \
#     --num-batch 7 \
#     --mem-constraint 18 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 1 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_7_bucketing_h_256.log
# echo '---start backpack_29_mem_30_split.py 8 batches '
# python arxiv_backpack_29_mem_30_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_30_backpack_bucketing \
#     --num-batch 8 \
#     --mem-constraint 18 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_8_bucketing_h_256.log
# echo '---start backpack_29_mem_30_split.py 9 batches '
# python arxiv_backpack_29_mem_30_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_30_backpack_bucketing \
#     --num-batch 9 \
#     --mem-constraint 18 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_9_bucketing_h_256.log
# echo '---start backpack_29_mem_30_split.py 10 batches '
# python arxiv_backpack_29_mem_30_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_30_backpack_bucketing \
#     --num-batch 10 \
#     --mem-constraint 18 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_10_bucketing_h_256.log
# echo '---start backpack_29_mem_30_split.py 12 batches '
# python arxiv_backpack_29_mem_30_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_30_backpack_bucketing \
#     --num-batch 12 \
#     --mem-constraint 18 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 1 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_12_bucketing_h_256.log
echo '---start backpack_29_mem_30_split.py 11 batches '
python arxiv_backpack_29_mem_30_split.py \
    --dataset ogbn-arxiv \
    --selection-method arxiv_30_backpack_bucketing \
    --num-batch 11 \
    --mem-constraint 18 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 256 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_11_bucketing_h_256.log