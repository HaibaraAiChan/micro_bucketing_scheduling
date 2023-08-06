#!/bin/bash
save_path=./betty_log/
hidden=128
md=range
save_path=./betty_log/${md}
for nb in 16 24 32
do
    echo "---start Betty_products_e2e.py hidden ${hidden},  nb ${nb} batches"
    python Betty_products_e2e.py  \
        --dataset ogbn-products \
        --selection-method $md \
        --num-batch ${nb} \
        --num-layers 2 \
        --fan-out 10,25 \
        --num-hidden ${hidden} \
        --num-runs 1 \
        --num-epoch 10 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
        > ${save_path}/${md}_nb_${nb}_e2e_${hidden}.log
done