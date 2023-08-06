#!/bin/bash

# mkdir ./log1

mkdir $save_path
hidden=2048
md=metis
n_layer=3
fanout=10,25,30
save_path=./betty_log/3-layer
data=cora
# for nb in 5 6 7 8 9 10 11 12 16 32
for nb in 2 3 4 5 6 
do
    echo "---start ${md} ${data}  ${nb} batches "
    python Betty_cora_e2e.py \
        --dataset $data \
        --selection-method $md \
        --num-batch $nb \
        --num-layers $n_layer \
        --fan-out $fanout\
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 5 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
        > ${save_path}/${md}_nb_${nb}_hidden_${hidden}_fanout_${fanout}.log
done

