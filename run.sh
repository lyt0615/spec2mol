# !/bin/bash
# 选择一组编号连续的GPU：--gpu_ids a-d；手动指定：--gpu_ids a,b,c,d...
# ***尽量用编号连续的，两边不要都有空余，例如0-4，或者4-7（7是最后一个）***
python /home/lyt/projects/spec2mol/main.py \
 --net Transformer \
 --mode test \
 --ds ir2mol \
 --gpu_ids 1-7 \
 --test_checkpoint checkpoints/ir2mol/Transformer/2025-10-30_00_20/510_acc_0361.pth \
 --device cuda:7 \
 --to_smarts

