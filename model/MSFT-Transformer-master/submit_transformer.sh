#!/bin/bash
#SBATCH --job-name=model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --gres=gpu:1
#SBATCH --output=transformer_%j.out
#SBATCH --error=transformer_%j.err

# 先加载 conda 的初始化脚本，然后使用 conda activate
# source /home/liang/miniconda3/etc/profile.d/conda.sh  # PT5
source /root/miniconda/etc/profile.d/conda.sh  # lanyunGPU
conda activate hj_env

python -u main.py -d EW-T2D -f ko,species -m MTMFTransformer -bs 16 -lr 1e-4 --gpu 0 -num_b 4 -n 2 -ub -uca
# python -u main.py -d EW-T2D -f ko,species -m MTMFTransformer -bs 16 -lr 1e-4 --gpu 0 -num_b 4 -ub -uca
# python -u main.py -d Obesity -f ko,species -m MTMFTransformer -bs 16 -lr 1e-4 --gpu 0 -num_b 4 -ub -uca
# python -u main.py -d AD -f ko,species -m MTMFTransformer -bs 16 -lr 1e-4 --gpu 0 -num_b 4 -ub -uca
# python -u main.py -d EW-T2D -f ko,species -m MBT -bs 4 -lr 6e-5 --gpu 0 -num_b 4 --n_layers 3 --m_layers 1 --hidden_size 0

# python -u main.py -d EW-T2D -f species -m FT -bs 16 --n_blocks 2 -lr 1e-4 --gpu 0 
