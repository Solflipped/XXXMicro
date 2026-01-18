#!/bin/bash
#SBATCH --job-name=humann3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --partition=parallel_vip_24h
#SBATCH --qos=parallel_vip_24h
#SBATCH --output=humann3_%j.out
#SBATCH --error=humann3_%j.err

# 先加载 conda 的初始化脚本，然后使用 conda activate
source /data/home/chenliang/apps/miniconda3/etc/profile.d/conda.sh
conda activate snakemake_env

snakemake \
  --snakefile ./humann3.snakefile \
  --configfile ./config_humann3.yaml \
  --use-conda \
  --conda-prefix ${HOME}/.conda/snakemake \
  --resources mem_mb=128000 \
  --cores $SLURM_CPUS_PER_TASK 
# --jobs 1 主要用于设置kraken2的snakefile 同一时间只运行一个 Kraken2 任务，避免多份数据库占用内存