#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
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
python cora_backpack.py \
    --dataset cora \
    --selection-method cora_30_backpack_bucketing \
    --num-batch 4 \
    --mem-constraint 3.75 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 2048 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_4_bucketing_h_2048_3.75.log