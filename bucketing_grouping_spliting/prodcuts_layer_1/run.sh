#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
mkdir $save_path
hidden=256
nb=2
# echo '---start backpack.py   2 batches '
# python backpack.py \
#     --dataset ogbn-products \
#     --selection-method products_20_backpack_bucketing \
#     --num-batch $nb \
#     --mem-constraint 18.1 \
#     --num-layers 1 \
#     --fan-out 20 \
#     --num-hidden $hidden \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_${nb}_1_layer.log
hidden=256
fanout=800
nb=13
echo '---start backpack.py   13 batches '
python backpack.py \
    --dataset ogbn-products \
    --selection-method products_${fanout}_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 18.1 \
    --num-layers 1 \
    --fan-out $fanout \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 2 \
    --lr 1e-2 \
    > ${save_path}/nb_${nb}_1_layer.log
