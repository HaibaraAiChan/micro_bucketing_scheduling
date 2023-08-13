#!/bin/bash
save_path=./betty_log/
hidden=1
md=range
save_path=./betty_log/${md}
data=reddit
# for nb in 16 24 32 17 18 19 20 21 22
for nb in  53 52
do
    echo "---start Betty_e2e.py hidden ${hidden},  nb ${nb} batches"
    python Betty_e2e.py  \
        --dataset $data \
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