#!/bin/bash
np=10
hidden=32
lr=1e-3
dataset=ogbn-products
datasetname=products

capacity=18
method=${datasetname}_25_backpack_bucketing
save_path=./Buckey_GAT_log/$datasetname

for nb in 18 19 20 22 24 
do
    python buckey_gat.py \
        --dataset $dataset \
        --selection-method $method \
        --num-batch $nb \
        --mem-constraint $capacity\
        --num-layers 2 \
        --fan-out 10,25 \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch $np \
        --aggre lstm \
        --log-indent 0 \
        --lr $lr \
        > ${save_path}/buckey_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
done







# np=10
# hidden=128
# lr=1e-3
# dataset=ogbn-arxiv
# datasetname=arxiv

# capacity=18
# method=${datasetname}_25_backpack_bucketing
# save_path=./Buckey_GAT_log/$datasetname

# for nb in 18 
# do
#     python buckey_gat.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --mem-constraint $capacity\
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 2 \
#         --lr $lr \
#         > ${save_path}/buckey_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done






# np=10
# hidden=1024
# lr=1e-3
# dataset=pubmed
# datasetname=pubmed

# capacity=18
# method=${datasetname}_25_backpack_bucketing
# save_path=./Buckey_GAT_log/$datasetname

# for nb in 7 8 
# do
#     python buckey_gat.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --mem-constraint $capacity\
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 2 \
#         --lr $lr \
#         > ${save_path}/buckey_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done



# np=10
# hidden=1024
# lr=1e-3
# dataset=cora
# datasetname=cora

# capacity=18
# method=${datasetname}_25_backpack_bucketing
# save_path=./Buckey_GAT_log/$datasetname

# for nb in 7
# do
#     python buckey_gat.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --mem-constraint $capacity\
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 2 \
#         --lr $lr \
#         > ${save_path}/buckey_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done




# np=10
# hidden=256
# lr=1e-3

# # dataset=cora
# # dataset=pubmed
# # dataset=ogbn-arxiv
# # datasetname=arxiv
# dataset=reddit
# datasetname=reddit


# method=${datasetname}_10_backpack_bucketing
# capacity=18
# save_path=./Buckey_GAT_log/$datasetname
# # mkdir $save_path

# # for nb in  16  18  20 22 24 26 32
# for nb in 16 18
# do
#     python buckey_gat.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --mem-constraint $capacity\
#         --num-layers 1 \
#         --fan-out 10 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 2 \
#         --lr $lr \
#         > ${save_path}/buckey_GAT_1_layer_h_${hidden}_nb_${nb}_np_${np}.log
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


# method=${datasetname}_25_backpack_bucketing
# capacity=18
# save_path=./Buckey_GAT_log/$datasetname
# # mkdir $save_path

# # for nb in  16  18  20 22 24 26 32
# for nb in  40
# do
#     python buckey_gat.py \
#         --dataset $dataset \
#         --selection-method $method \
#         --num-batch $nb \
#         --mem-constraint $capacity\
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch $np \
#         --aggre lstm \
#         --log-indent 2 \
#         --lr $lr \
#         > ${save_path}/buckey_GAT_2_layer_h_${hidden}_nb_${nb}_np_${np}.log
# done
