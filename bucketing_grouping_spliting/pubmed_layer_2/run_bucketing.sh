#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
# mkdir $save_path
nb=1
hidden=256
np=400
# echo "---start pubmed_backpack.py ${nb} batches "
# python pubmed_backpack.py \
#     --dataset pubmed \
#     --selection-method pubmed_25_backpack_bucketing \
#     --num-batch $nb \
#     --mem-constraint 0.12 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden $hidden \
#     --num-runs 1 \
#     --num-epoch $np \
#     --aggre lstm \
#     --log-indent 0 \
#     --lr 1e-3 \
#     > ${save_path}/nb_${nb}_np_${np}_bucketing_h_${hidden}.log
nb=2
# np=4
echo "---start pubmed_backpack.py ${nb} batches "
python pubmed_backpack.py \
    --dataset pubmed \
    --selection-method pubmed_25_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 0.045 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch $np \
    --aggre lstm \
    --log-indent 0 \
    --lr 1e-3 \
    > ${save_path}/nb_${nb}_np_${np}_bucketing_h_${hidden}.log
nb=4
echo "---start pubmed_backpack.py ${nb} batches "
python pubmed_backpack.py \
    --dataset pubmed \
    --selection-method pubmed_25_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 0.023 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch $np \
    --aggre lstm \
    --log-indent 0 \
    --lr 1e-3 \
    > ${save_path}/nb_${nb}_np_${np}_bucketing_h_${hidden}.log
# echo '---start pubmed_backpack.py 3 batches '
# python pubmed_backpack.py \
#     --dataset pubmed \
#     --selection-method pubmed_30_backpack_bucketing \
#     --num-batch 2 \
#     --mem-constraint 7.82 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 4096 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-5 \
#     > ${save_path}/nb_3_bucketing_h_4096.log
# echo '---start pubmed_backpack.py 3 batches '
# python pubmed_backpack.py \
#     --dataset pubmed \
#     --selection-method pubmed_30_backpack_bucketing \
#     --num-batch 3 \
#     --mem-constraint 5.4 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 4096 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-5 \
#     > ${save_path}/nb_3_bucketing_h_4096.log
# echo '---start pubmed_backpack.py 4 batches '
# python pubmed_backpack.py \
#     --dataset pubmed \
#     --selection-method pubmed_30_backpack_bucketing \
#     --num-batch 4 \
#     --mem-constraint 4 \
#     --num-layers 3 \
#     --fan-out 10,25,30 \
#     --num-hidden 4096 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-5 \
#     > ${save_path}/nb_4_bucketing_h_4096.log
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