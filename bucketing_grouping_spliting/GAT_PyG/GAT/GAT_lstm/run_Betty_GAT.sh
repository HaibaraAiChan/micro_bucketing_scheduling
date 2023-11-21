#!/bin/bash

np=10
hidden=256
lr=1e-3

# dataset=cora
# dataset=pubmed
dataset=ogbn-arxiv
datasetname=arxiv
# dataset=reddit
# datasetname=reddit

method=REG
save_path=./Betty_GAT_log/$datasetname
# mkdir $save_path

for nb in  14 15 16  18  20 24 32
do
    python Betty_GAT.py \
        --dataset $dataset \
        --selection-method $method \
        --num-batch $nb \
        --num-layers 2 \
        --fan-out 10,25 \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch $np \
        --aggre lstm \
        --log-indent 3 \
        --lr $lr \
        > ${save_path}/betty_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
done
