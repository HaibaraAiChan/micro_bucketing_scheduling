#!/bin/bash

mkdir ./log1
save_path=./log1/betty
mkdir $save_path
# echo '---start Betty_time.py REG  9 batches '
# python Betty_time.py \
#     --dataset ogbn-products \
#     --num-batch 9 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --lr 1e-2 \
#     > ${save_path}/nb_9_betty.log

# echo '---start Betty_time.py REG  8 batches '
# python Betty_time.py \
#     --dataset ogbn-products \
#     --num-batch 8 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --lr 1e-2 \
#     > ${save_path}/nb_8_betty.log
# echo '---start Betty_time.py REG  7 batches '
# python Betty_time.py \
#     --dataset ogbn-products \
#     --num-batch 7 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --lr 1e-2 \
#     > ${save_path}/nb_7_betty.log

# echo '---start Betty_time.py REG  10 batches '
# python Betty_time.py \
#     --dataset ogbn-products \
#     --num-batch 10 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --lr 1e-2 \
#     > ${save_path}/nb_10_betty.log

echo '---start Betty_time.py REG  12 batches '
python Betty_time.py \
    --dataset ogbn-products \
    --num-batch 12 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 128 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --lr 1e-2 \
    > ${save_path}/nb_12_betty.log