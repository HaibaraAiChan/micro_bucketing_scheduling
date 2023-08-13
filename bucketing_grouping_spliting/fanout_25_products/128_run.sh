#!/bin/bash
save_path=./bucketing_log/128
hidden=128
np=1
# for nb in  15 16 17 18 19 20 32
# for nb in  18 19 20 24 32 
for nb in  24
do
    echo "---start products_25_time.py   nb  ${nb}  batches "
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
        --log-indent 3 \
        --lr 1e-2 \
        > ${save_path}/nb_${nb}_bucketing_${hidden}_new.log
done
# echo '---start products_25_time.py REG  15 batches '
# python products_25_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_products_bucketing \
#     --num-batch 15 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_15_bucketing.log
# echo '---start products_25_time.py REG  14 batches '
# python products_25_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_products_bucketing \
#     --num-batch 14 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
#     > ${save_path}/nb_14_bucketing_pybind11.log
# echo '---start products_25_time.py REG  13 batches '
# python products_25_time.py \
#     --dataset ogbn-products \
#     --selection-method 25_backpack_products_bucketing \
#     --num-batch 13 \
#     --mem-constraint 18.1 \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden 128 \
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
#     --num-hidden 128 \
#     --num-runs 1 \
#     --num-epoch 20 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-2 \
# > ${save_path}/nb_12_bucketing_pybind11_20_epochs.log
