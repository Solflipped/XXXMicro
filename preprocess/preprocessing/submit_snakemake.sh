#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --partition=parallel_vip_24h
#SBATCH --qos=parallel_vip_24h
#SBATCH --output=preprocessing_%j.out
#SBATCH --error=preprocessing_%j.err

# 先加载 conda 的初始化脚本，然后使用 conda activate
source /data/home/chenliang/apps/miniconda3/etc/profile.d/conda.sh
conda activate snakemake_env

snakemake \
  --snakefile ./preprocessing.snakefile \
  --configfile ./config_preprocessing.yaml \
  --use-conda \
  --conda-prefix ${HOME}/.conda/snakemake \
  --cores $SLURM_CPUS_PER_TASK 