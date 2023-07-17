#!/bin/bash

# mkdir ./log1
save_path=./betty_log
# mkdir $save_path

echo '---start Betty_cora_e2e_10_25.py 2 batches '
python Betty_cora_e2e_10_25.py \
    --dataset cora \
    --selection-method REG \
    --num-batch 1 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 2048 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_1_h_2048.log
echo '---start Betty_cora_e2e_10_25.py 2 batches '
python Betty_cora_e2e_10_25.py \
    --dataset cora \
    --selection-method REG \
    --num-batch 2 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden 2048 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/nb_2_h_2048.log
