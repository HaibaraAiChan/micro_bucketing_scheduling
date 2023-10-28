#!/bin/bash

# mkdir ./log1
save_path=./betty_log
# mkdir $save_path

# echo '---start Betty_cora_e2e_10_25.py 1 batches '
# python Betty_cora_e2e_10_25.py \
#     --dataset cora \
#     --selection-method REG \
#     --num-batch 1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/train_loss_nb_1_h_256.log

method=REG
nb=1
dataset=karate
fanout=2,4
hidden=1024
echo "---start Betty_cora_e2e_10_25.py ${nb} batches $method"
python Betty_cora_e2e_10_25.py \
    --dataset $dataset \
    --selection-method $method \
    --num-batch $nb \
    --num-layers 2 \
    --fan-out $fanout \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    > ${save_path}/${dataset}_train_loss_nb_${nb}_h_${hidden}_${method}.log
# echo '---start Betty_cora_e2e_10_25.py 2 batches '
# python Betty_cora_e2e_10_25.py \
#     --dataset cora \
#     --selection-method REG \
#     --num-batch 2 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/train_loss_nb_2_h_256.log
