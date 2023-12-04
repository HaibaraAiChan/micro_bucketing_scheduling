#!/bin/bash

# mkdir ./log1


hidden=128
md=REG
n_layer=2
fanout=10,25
save_path=./betty_log/2-layer/betty_slow_version
mkdir $save_path
for nb in 18
do
    echo "---start ${md}  ${nb} batches "
    python Betty_arxiv_e2e.py \
        --dataset ogbn-arxiv \
        --selection-method $md \
        --num-batch $nb \
        --num-layers $n_layer \
        --fan-out $fanout\
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 15 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
        > ${save_path}/${md}_nb_${nb}_hidden_${hidden}_fanout_${fanout}.log
done

