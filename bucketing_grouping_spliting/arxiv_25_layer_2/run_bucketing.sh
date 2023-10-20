#!/bin/bash

# mkdir ./log1
# save_path=./bucketing_log/1024
# mkdir $save_path
# echo '---start backpack_24_mem_25_split.py   4 batches '
# python arxiv_backpack_24_mem_25_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_25_backpack_bucketing \
#     --num-batch 4 \
#     --mem-constraint 18 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 1024 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_4_bucketing_1024_new.log
# echo '---start backpack_24_mem_25_split.py  5 batches '
# python arxiv_backpack_24_mem_25_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_25_backpack_bucketing \
#     --num-batch 5 \
#     --mem-constraint 18 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 1024 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_5_bucketing_1024_pybind11.log
# echo '---start backpack_24_mem_25_split.py  6 batches '
# python arxiv_backpack_24_mem_25_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_25_backpack_bucketing \
#     --num-batch 6 \
#     --mem-constraint 18 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 1024 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_6_bucketing_1024___.log
# nb=32
# echo '---start backpack_24_mem_25_split.py {$nb} batches '
# python arxiv_backpack_24_mem_25_split.py \
#     --dataset ogbn-arxiv \
#     --selection-method arxiv_25_backpack_bucketing \
#     --num-batch $nb \
#     --mem-constraint 18 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 1024 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_${nb}_h_1024_bk_2.2.log

save_path=./bucketing_log/256
nb=16
hidden=256
n_epoch=1
echo "---start backpack_24_mem_25_split.py ${nb} batches "
python arxiv_backpack_24_mem_25_split.py \
    --dataset ogbn-arxiv \
    --selection-method arxiv_25_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 18 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch $n_epoch \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-3 \
    --eval-every 5 \
    > ${save_path}/nb_${nb}_h_${hidden}_epoch_${n_epoch}_group.log