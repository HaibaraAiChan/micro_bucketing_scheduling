#!/bin/bash

# mkdir ./log1

mkdir $save_path
hidden=602

n_layer=1
fanout=10
np=10
data=reddit

for md in random range metis
do
    save_path=./betty_log/${md}
    for nb in 3 4 5 6 7 
    do
        echo "---start ${md}  ${data}  ${nb} batches "
        python Betty_reddit_e2e.py \
            --dataset $data \
            --selection-method $md \
            --num-batch $nb \
            --num-layers $n_layer \
            --fan-out $fanout\
            --num-hidden $hidden \
            --num-runs 1 \
            --num-epoch $np \
            --aggre lstm \
            --log-indent 3 \
            --lr 1e-3 \
            > ${save_path}/nb_${nb}_hidden_${hidden}_fanout_${fanout}.log
    done
done
