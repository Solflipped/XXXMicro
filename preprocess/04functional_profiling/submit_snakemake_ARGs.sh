#!/bin/bash
#SBATCH --job-name=ARGs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --partition=parallel_vip_24h
#SBATCH --qos=parallel_vip_24h
#SBATCH --output=ARGs_%j.out
#SBATCH --error=ARGs_%j.err

# 先加载 conda 的初始化脚本，然后使用 conda activate
source /data/home/chenliang/apps/miniconda3/etc/profile.d/conda.sh
conda activate snakemake_env

snakemake \
  --snakefile ./ARGs.snakefile \
  --configfile ./config_ARGs.yaml \
  --use-conda \
  --conda-prefix ${HOME}/.conda/snakemake \
  --cores $SLURM_CPUS_PER_TASK 