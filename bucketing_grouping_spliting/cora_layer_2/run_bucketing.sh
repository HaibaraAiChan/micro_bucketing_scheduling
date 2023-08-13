#!/bin/bash

# mkdir ./log1
save_path=./bucket_log/256
# mkdir $save_path

# echo '---start cora_backpack.py 3 batches '
# python cora_backpack.py \
#     --dataset cora \
#     --selection-method cora_30_backpack_bucketing \
#     --num-batch 3 \
#     --mem-constraint 4.8 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 2048 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-5 \
#     > ${save_path}/nb_3_bucketing_h_2048.log
# echo '---start cora_backpack.py 4 batches '
# python cora_backpack.py \
#     --dataset cora \
#     --selection-method cora_30_backpack_bucketing \
#     --num-batch 4 \
#     --mem-constraint 3.8 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 2048 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-5 \
#     > ${save_path}/nb_4_bucketing_h_2048.log
# echo '---start cora_backpack.py 4 batches '
nb=4
hidden=256
python cora_backpack.py \
    --dataset cora \
    --selection-method cora_25_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 0.095 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch 1 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_${nb}_bucketing_h_${hidden}___.log