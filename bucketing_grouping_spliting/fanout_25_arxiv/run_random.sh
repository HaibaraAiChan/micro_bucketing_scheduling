#!/bin/bash

# mkdir ./log1

mkdir $save_path
hidden=1024
# md=random
# md=range
md=metis
n_layer=2
fanout=10,25
save_path=./betty_log/2-layer/${md}
for nb in 5 6 7 8 9 10 11 12 16 32
do
    echo "---start ${md}   ${nb} batches "
    python Metis.py \
        --dataset ogbn-arxiv \
        --selection-method $md \
        --num-batch $nb \
        --num-layers $n_layer \
        --fan-out $fanout\
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 10 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
        > ${save_path}/${md}_nb_${nb}_hidden_${hidden}_fanout_${fanout}.log
done

