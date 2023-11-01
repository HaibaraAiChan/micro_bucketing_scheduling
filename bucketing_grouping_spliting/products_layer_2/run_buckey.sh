#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
mkdir $save_path
hidden=256
nb=4
np=5

echo "---start backpack.py  ${nb} batches "
python backpack.py \
    --dataset ogbn-products \
    --selection-method products_25_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 18.1 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch $np \
    --aggre lstm \
    --log-indent 0 \
    --lr 1e-2 \
    > ${save_path}/${np}_epoch_nb_${nb}_2_layer_h_${hidden}.log
# hidden=256
# fanout=800
# nb=13
# echo '---start backpack.py   13 batches '
# python backpack.py \
#     --dataset ogbn-products \
#     --selection-method products_${fanout}_backpack_bucketing \
#     --num-batch $nb \
#     --mem-constraint 18.1 \
#     --num-layers 1 \
#     --fan-out $fanout \
#     --num-hidden $hidden \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 2 \
#     --lr 1e-2 \
#     > ${save_path}/nb_${nb}_1_layer.log
# hidden=256
# fanout=10
# nb=2
# echo "---start backpack.py  ${nb} batches "
# python backpack.py \
#     --dataset ogbn-products \
#     --selection-method products_${fanout}_backpack_bucketing \
#     --num-batch $nb \
#     --mem-constraint 18.1 \
#     --num-layers 1 \
#     --fan-out $fanout \
#     --num-hidden $hidden \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 2 \
#     --lr 1e-2 \
#     > ${save_path}/nb_${nb}_fanout_${fanout}_1_layer.log