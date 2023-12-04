#!/bin/bash

# np=10
# hidden=128
# lr=1e-3

# # dataset=cora
# # dataset=pubmed
# dataset=ogbn-arxiv
# datasetname=arxiv
# # dataset=reddit
# # datasetname=reddit

# method=REG
# save_path=./Betty_GAT_log/$datasetname
# # mkdir $save_path

# for nb in   4 5 6 7 8 10 14
# do
#     python Betty_GAT.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 0 \
#         --lr $lr \
#         > ${save_path}/betty_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done


# np=10
# hidden=256
# lr=1e-3

# # dataset=cora
# # dataset=pubmed
# dataset=ogbn-arxiv
# datasetname=arxiv
# # dataset=reddit
# # datasetname=reddit

# method=REG
# save_path=./Betty_GAT_log/$datasetname
# # mkdir $save_path

# for nb in  14 15 16  18  20 24 32
# do
#     python Betty_GAT.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 3 \
#         --lr $lr \
#         > ${save_path}/betty_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done

# np=10
# hidden=256
# lr=1e-3
# dataset=reddit
# datasetname=reddit

# method=REG
# save_path=./Betty_GAT_log/$datasetname
# # mkdir $save_path

# for nb in  4  6  8 10 12 14 16 18 
# do
#     python Betty_GAT.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --num-layers 1 \
#         --fan-out 10 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 3 \
#         --lr $lr \
#         > ${save_path}/betty_GAT_1_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done

# np=10
# hidden=1024
# lr=1e-3
# dataset=cora
# datasetname=cora

# method=REG
# save_path=./Betty_GAT_log/$datasetname
# # mkdir $save_path

# for nb in  3
# do
#     python Betty_GAT.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 3 \
#         --lr $lr \
#         > ${save_path}/betty_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done


# np=5
# hidden=1024
# lr=1e-3
# dataset=pubmed
# datasetname=pubmed

# method=REG
# save_path=./Betty_GAT_log/$datasetname
# # mkdir $save_path

# for nb in  10  12  14  16 32
# do
#     python Betty_GAT.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 0 \
#         --lr $lr \
#         > ${save_path}/betty_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done


np=10
hidden=16
lr=1e-3
dataset=ogbn-products
datasetname=products

method=REG
save_path=./Betty_GAT_log/$datasetname
# mkdir $save_path

for nb in 4 6 8 10  12  14  16 
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
        --log-indent 0 \
        --lr $lr \
        > ${save_path}/betty_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
done