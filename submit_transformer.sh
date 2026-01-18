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
source /data/home/chenliang/apps/miniconda3/etc/profile.d/conda.sh
conda activate hj_env

module load cuda
nvidia-smi
python test_pytorch.py

# AD
# python -u main.py -d AD -f ko,species -m MBT -bs 8 -lr 1e-4 --gpu 0 -fl 3  -num_b 4 -nb 6
# python -u main.py -d AD -f ko,species -m MBT -bs 16 -lr 1e-4 --gpu 0 -fl 2  -num_b 4 -nb 4
# python -u main.py -d AD -f species -m FT_transformer -bs 16 -lr 1e-4 --gpu 0 -nb 4
# python -u main.py -d AD -f ko,species -m MDL4Microbiome -bs 16 -lr 1e-4 --gpu 0 
# python -u main.py -d AD -f ko,species -m MSFTTransformer -bs 8 -lr 1e-4 --gpu 0 -num_b 4 -nb 4
# python -u main.py -d AD -f ko,species -m FT_Vote -bs 8 -lr 1e-4 --gpu 0 -nb 4
# python -u main.py -d AD -f ko,species -m GAFT -bs 8 -nb 4 -fl 2 -num_b 4 -lr 1e-4 --gpu 0 --finetune_mbt
python main.py -d AD -f ko,species -m GAFT \
  --n_blocks 4 \
  --fusion_layer 2 \
  --num_bottleneck 4 \
  --mbt_use_cross_atn \
  --lmf_hidden_dim 128 \
  --lmf_output_dim 256 \
  --lmf_rank 4 \
  --use_lmf_subnet \
  --gat_dim 256 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --finetune_mbt


# EW-T2D
# python -u main.py -d EW-T2D -f ko,species -m MBT -bs 8 -lr 1e-4 --gpu 0 -fl 2  -num_b 4 -nb 4
# python -u main.py -d EW-T2D -f ko,species -m MSFTTransformer -bs 8 -lr 1e-4 --gpu 0 -num_b 4 -nb 4
# python -u main.py -d EW-T2D -f ko,species -m FT_Vote -bs 8 -lr 1e-4 --gpu 0 -nb 4
# python -u main.py -d EW-T2D -f species -m FT_transformer -bs 8 -lr 1e-4 --gpu 0 -nb 6
# python -u main.py -d EW-T2D -f ko,species -m GAFT -bs 8 -nb 4 -fl 2 -num_b 4 -lr 1e-4 --gpu 0 --finetune_mbt
# python main.py -d EW-T2D -f ko,species -m GAFT \
#   --n_blocks 4 \
#   --fusion_layer 2 \
#   --num_bottleneck 4 \
#   --mbt_use_cross_atn \
#   --lmf_hidden_dim 128 \
#   --lmf_output_dim 256 \
#   --lmf_rank 4 \
#   --use_lmf_subnet \
#   --gat_dim 256 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --finetune_mbt


# Obesity
# python -u main.py -d Obesity -f ko,species -m MBT -bs 8 -lr 1e-4 --gpu 0 -fl 3  -num_b 4 -nb 6
# python -u main.py -d AD -f species -m FT_transformer -bs 8 -lr 1e-4 --gpu 0 -nb 4
# python -u main.py -d AD -f ko,species -m MDL4Microbiome -bs 8 -lr 1e-4 --gpu 0 -e1 30 -e2 20

