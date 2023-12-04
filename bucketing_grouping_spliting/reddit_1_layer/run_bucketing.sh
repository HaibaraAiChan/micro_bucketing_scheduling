#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
# mkdir $save_path
nb=8
echo "---start reddit_backpack.py   ${nb} batches "
python reddit_backpack.py \
    --dataset reddit \
    --selection-method reddit_10_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 18 \
    --num-layers 1 \
    --fan-out 10 \
    --num-hidden 602 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-4 \
    > ${save_path}/nb_${nb}_bucketing.log
# echo '---start reddit_backpack.py   4 batches '
# python reddit_backpack.py \
#     --dataset reddit \
#     --selection-method reddit_10_backpack_bucketing \
#     --num-batch 4 \
#     --mem-constraint 18 \
#     --num-layers 1 \
#     --fan-out 10 \
#     --num-hidden 602 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-4 \
#     > ${save_path}/nb_4_bucketing__.log
# echo '---start reddit_backpack.py  5 batches '
# python reddit_backpack.py \
#     --dataset reddit \
#     --selection-method reddit_10_backpack_bucketing \
#     --num-batch 5 \
#     --mem-constraint 18 \
#     --num-layers 1 \
#     --fan-out 10 \
#     --num-hidden 602 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-4 \
#     > ${save_path}/nb_5_bucketing.log
# echo '---start reddit_backpack.py  6 batches '
# python reddit_backpack.py \
#     --dataset reddit \
#     --selection-method reddit_10_backpack_bucketing \
#     --num-batch 6 \
#     --mem-constraint 18 \
#     --num-layers 1 \
#     --fan-out 10 \
#     --num-hidden 602 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-4 \
#     > ${save_path}/nb_6_bucketing.log