#!/bin/bash

# mkdir ./log1
save_path=./bucketing_log
mkdir $save_path
nb=51
hidden=256
np=40
echo "---start backpack_24_mem_25_split.py ${nb} batches "
python reddit_backpack.py \
    --dataset reddit \
    --selection-method reddit_25_backpack_bucketing \
    --num-batch $nb \
    --mem-constraint 18 \
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch $np \
    --aggre lstm \
    --log-indent 0 \
    --lr 1e-2 \
    > ${save_path}/${np}_epoch_nb_${nb}_2_layer_h_${hidden}.log