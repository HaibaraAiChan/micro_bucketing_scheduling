#!/bin/bash

# mkdir ./log1

mkdir $save_path
hidden=128
# md=random
md=range
# md=metis
n_layer=2
fanout=10,25
save_path=./betty_log/2-layer/${md}
for nb in  2 4 8 10 12 14 16 18
do
    echo "---start ${md}   ${nb} batches "
    python Betty_arxiv_e2e.py \
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

