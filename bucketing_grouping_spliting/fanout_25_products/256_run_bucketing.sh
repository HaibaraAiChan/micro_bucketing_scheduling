#!/bin/bash

# mkdir ./log1
# save_path=../log/10_epoch/bucketing_optimized
save_path=./bucketing_log/256
# mkdir $save_path
# echo '---start products_25_time.py REG  16 batches '
# python products_25_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_products_bucketing \
#     --num-batch 16 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_16_bucketing.log
hidden=256
nb=15
np=60
echo "---start products_25_time.py REG  ${nb} batches "
python products_25_time.py \
    --dataset ogbn-products \
    --selection-method 25_backpack_products_bucketing \
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
# echo '---start products_25_time.py REG  14 batches '
# python products_25_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_products_bucketing \
#     --num-batch 14 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_14.log
# echo '---start products_25_time.py REG  13 batches '
# python products_25_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_products_bucketing \
#     --num-batch 13 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_13_bucketing_pybind11.log

# echo '---start products_25_time.py REG  12 batches '
# python products_25_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_products_bucketing \
#     --num-batch 12 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 256 \
#     --num-runs 1 \
#     --num-epoch 20 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
# > ${save_path}/nb_12_bucketing_pybind11_20_epochs.log
