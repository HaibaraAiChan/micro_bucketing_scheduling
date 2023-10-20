#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
# mkdir $save_path
echo '---start pubmed_backpack.py 2 batches '
python pubmed_backpack.py \
    --dataset pubmed \
    --selection-method pubmed_30_backpack_bucketing \
    --num-batch 2 \
    --mem-constraint 7.82 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 4096 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_2_bucketing_h_4096.log

echo '---start pubmed_backpack.py 3 batches '
python pubmed_backpack.py \
    --dataset pubmed \
    --selection-method pubmed_30_backpack_bucketing \
    --num-batch 3 \
    --mem-constraint 5.4 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 4096 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_3_bucketing_h_4096.log
echo '---start pubmed_backpack.py 4 batches '
python pubmed_backpack.py \
    --dataset pubmed \
    --selection-method pubmed_30_backpack_bucketing \
    --num-batch 4 \
    --mem-constraint 4 \
    --num-layers 3 \
    --fan-out 10,25,30 \
    --num-hidden 4096 \
    --num-runs 1 \
    --num-epoch 10 \
    --aggre lstm \
    --log-indent 3 \
    --lr 1e-5 \
    > ${save_path}/nb_4_bucketing_h_4096.log
# echo '---start pubmed_backpack.py 4 batches '
# python pubmed_backpack.py \
#     --dataset pubmed \
#     --selection-method pubmed_30_backpack_bucketing \
#     --num-batch 4 \
#     --mem-constraint 3.75 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 4096 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-5 \
#     > ${save_path}/nb_4_bucketing_h_4096_3.75.log