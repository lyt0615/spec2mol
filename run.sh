# !/bin/bash
# SBATCH --job-name=GPU_HPL
# SBATCH --partition=pgpu
# SBATCH -n 4
# SBATCH --ntasks-per-node=4
# SBATCH --gres=gpu:4
# SBATCH --mail-type=end
# SBATCH --output=%j.out
# SBATCH --error=%j.err
# /home/fangyikai/anaconda3/envs/raman/bin/python /home/fangyikai/sub_id-main/1.py
# /home/fangyikai/anaconda3/envs/raman/bin/python /home/fangyikai/sub_id-main/main_ddp.py --net VanillaTransformer --pretrain --device cuda:0 --test_checkpoint checkpoints/ir/MLPMixer/2025-07-25_09_13/421_f1_5592.pth --ds ir_pretrain --depth 1 --use_mixer 1 --use_se 1 --n_mixer 2
/home/fangyikai/anaconda3/envs/raman/bin/python -m pdb /home/fangyikai/sub_id-main/main.py --net MLPMixer --train --test_checkpoint checkpoints/ir/MLPMixer/2025-07-21_14_37/99_f1_5500.pth --ds ir --depth 1 --use_mixer 1 --use_se 1 --n_mixer 2
#-m torch.distributed.run --nproc_per_node 2 --master_port 29501