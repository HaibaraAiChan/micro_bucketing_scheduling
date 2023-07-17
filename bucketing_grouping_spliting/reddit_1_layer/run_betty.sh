#!/bin/bash

# mkdir ./log1
save_path=./betty_log
# mkdir $save_path
echo '---start Betty_reddit_e2e.py 3 batches '
python Betty_reddit_e2e.py \
    --dataset reddit \
    --selection-method REG \
    --num-batch 3 \
    --num-layers 1 \
    --fan-out 10 \
    --num-hidden 128 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-4 \
    > ${save_path}/nb_3_h_128.log
echo '---start Betty_reddit_e2e.py 4 batches '
python Betty_reddit_e2e.py \
    --dataset reddit \
    --selection-method REG \
    --num-batch 4 \
    --num-layers 1 \
    --fan-out 10 \
    --num-hidden 128 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-4 \
    > ${save_path}/nb_4_h_128.log
echo '---start Betty_reddit_e2e.py 5 batches '
python Betty_reddit_e2e.py \
    --dataset reddit \
    --selection-method REG \
    --num-batch 5 \
    --num-layers 1 \
    --fan-out 10 \
    --num-hidden 128 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-4 \
    > ${save_path}/nb_5_h_128.log
