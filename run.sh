# !/bin/bash
/home/lyt/anaconda3/envs/raman/bin/python /home/lyt/projects/spec2mol/main.py \
 --net Transformer \
 --mode test \
 --ds ir2mol \
 --gpu_ids 0,1,2,3 \
 --test_checkpoint checkpoints/ir2mol/Transformer/2025-09-23_20_45/470_acc_4604.pth \
 --device cuda:1 \
