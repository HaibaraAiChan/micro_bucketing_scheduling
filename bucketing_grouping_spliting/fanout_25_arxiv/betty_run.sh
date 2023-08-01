# #!/bin/bash

# # mkdir ./log1
# save_path=./betty_log
# mkdir $save_path
# # echo '---start Betty_arxiv_e2e.py   1 batches '
# # python Betty_arxiv_e2e.py  \
# #     --dataset ogbn-arxiv \
# #     --selection-method REG \
# #     --num-batch 1 \
# #     --num-layers 2 \
# #     --fan-out 10,25 \
# #     --num-hidden 128 \
# #     --num-runs 1 \
# #     --num-epoch 10 \
# #     --aggre lstm \
# #     --log-indent 3 \
# #     --lr 1e-3 \
# #     > ${save_path}/nb_1_e2e_128.log
# # echo '---start Betty_arxiv_e2e.py hidden 64,  1 batches '
# # python Betty_arxiv_e2e.py  \
# #     --dataset ogbn-arxiv \
# #     --selection-method REG \
# #     --num-batch 1 \
# #     --num-layers 2 \
# #     --fan-out 10,25 \
# #     --num-hidden 64 \
# #     --num-runs 1 \
# #     --num-epoch 10 \
# #     --aggre lstm \
# #     --log-indent 3 \
# #     --lr 1e-3 \
# #     > ${save_path}/nb_1_e2e_64.log
# hidden=1024
# nb=8
# echo '---start Betty_arxiv_e2e.py hidden {$hidden},  nb {$nb} batches '
# python Betty_arxiv_e2e.py  \
#     --dataset ogbn-arxiv \
#     --selection-method REG \
#     --num-batch $nb \
#     --num-layers 2 \
#     --fan-out 10,25 \
#     --num-hidden $hidden \
#     --num-runs 1 \
#     --num-epoch 10 \
#     --aggre lstm \
#     --log-indent 3 \
#     --lr 1e-3 \
#     > ${save_path}/nb_${nb}_e2e_${hidden}.log
#!/bin/bash
save_path=./betty_log
hidden=1024
# for nb in 9 10 11 12 16
for nb in 32
do
    echo "---start Betty_arxiv_e2e.py hidden ${hidden},  nb ${nb} batches"
    python Betty_arxiv_e2e.py  \
        --dataset ogbn-arxiv \
        --selection-method REG \
        --num-batch ${nb} \
        --num-layers 2 \
        --fan-out 10,25 \
        --num-hidden ${hidden} \
        --num-runs 1 \
        --num-epoch 10 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
        > ${save_path}/nb_${nb}_e2e_${hidden}.log
done
