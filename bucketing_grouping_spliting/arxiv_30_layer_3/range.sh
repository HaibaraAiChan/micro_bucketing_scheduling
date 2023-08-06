#!/bin/bash
save_path=./betty_log
hidden=1
n_layer=3
fanout=10,25,30
save_path=./betty_log/3-layer
md=range
# for nb in 9 10 11 12 16
for nb in 4
do
    echo "---start Betty_arxiv_e2e.py hidden ${hidden},  nb ${nb} batches"
    python Betty_arxiv_e2e.py  \
        --dataset ogbn-arxiv \
        --selection-method $md \
        --num-batch $nb \
        --num-layers $n_layer \
        --fan-out $fanout \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 5 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
        > ${save_path}/${md}_nb_${nb}_e2e_${hidden}_fanout_${fanout}.log
done
