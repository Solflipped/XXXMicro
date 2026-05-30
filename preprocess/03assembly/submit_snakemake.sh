#!/bin/bash
#SBATCH --job-name=assembly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=parallel_vip_1amd_12h
#SBATCH --qos=parallel_vip_1amd_12h
#SBATCH --output=assembly_%j.out
#SBATCH --error=assembly_%j.err

# 先加载 conda 的初始化脚本，然后使用 conda activate
source /data/home/chenliang/apps/miniconda3/etc/profile.d/conda.sh
conda activate snakemake_env

snakemake \
  --snakefile ./assembly.snakefile \
  --configfile ./config_assembly.yaml \
  --use-conda \
  --conda-prefix ${HOME}/.conda/snakemake \
  --cores $SLURM_CPUS_PER_TASK 