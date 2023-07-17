#!/bin/bash

# mkdir ./log1
save_path=./betty_log
# mkdir $save_path

echo '---start Betty_pubmed_e2e.py 1 batches '
python Betty_pubmed_e2e.py \
    --dataset pubmed \
    --selection-method REG \
    --num-batch 1 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 4096 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_1_h_4096.log
echo '---start Betty_pubmed_e2e.py 2 batches '
python Betty_pubmed_e2e.py \
    --dataset pubmed \
    --selection-method REG \
    --num-batch 2 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 4096 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_2_h_4096.log
echo '---start Betty_pubmed_e2e.py 3 batches '
python Betty_pubmed_e2e.py \
    --dataset pubmed \
    --selection-method REG \
    --num-batch 3 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 4096 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_3_h_4096.log
echo '---start Betty_pubmed_e2e.py 4 batches '
python Betty_pubmed_e2e.py \
    --dataset pubmed \
    --selection-method REG \
    --num-batch 4 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 4096 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_4_h_4096.log