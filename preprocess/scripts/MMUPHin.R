#!/usr/bin/env Rscript

# 批次效应矫正脚本
# 数据：ko_abundance.csv, species_abundance.csv, metadata_NC_AD.csv
# 批次变量：HPC_Batch, Group
# 协变量：Gender, Age
# 可视化分组：disease (AD/NC)
# 日期：2025-11-07

# 设置工作目录
setwd("/Users/jianhe/Desktop/snakemake/PRJCA022804")
dir.create("results", showWarnings = FALSE)

# 验证文件存在
meta_file <- "metadata_NC_AD.csv"
ko_file <- "ko_abundance.csv"
species_file <- "species_abundance.csv"
if (!file.exists(meta_file) || !file.exists(ko_file) || !file.exists(species_file)) {
  stop("以下文件缺失：",
       ifelse(file.exists(meta_file), "", paste(meta_file, " ")),
       ifelse(file.exists(ko_file), "", paste(ko_file, " ")),
       ifelse(file.exists(species_file), "", species_file))
}

# 1. 加载R包
p_list <- c("magrittr", "dplyr", "ggplot2", "vegan", "ggsci", "patchwork", "cowplot", "RColorBrewer")
for (p in p_list) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
  library(p, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
}
if (!requireNamespace("MMUPHin", quietly = TRUE)) {
  install.packages("devtools", repos = "https://cloud.r-project.org")
  devtools::install_github("biobakery/mmuphin@master")
}
library(MMUPHin)

# 2. 导入数据
meta.all <- read.csv(meta_file, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
meta.all$HPC_Batch <- factor(meta.all$HPC_Batch)
meta.all$Group <- factor(meta.all$Group)
meta.all$disease <- factor(meta.all$disease, levels = c("NC", "AD"))
meta.all$Gender <- factor(meta.all$Gender)
meta.all$Age <- as.numeric(meta.all$Age)  # 确保Age为数值型
meta.all$sample_id <- as.character(trimws(meta.all$sample_id))
rownames(meta.all) <- meta.all$sample_id  # 设置行名

# 验证元数据
required_cols <- c("sample_id", "HPC_Batch", "Group", "Gender", "Age", "disease")
if (!all(required_cols %in% colnames(meta.all))) {
  stop("元数据缺少必需列：", paste(required_cols[!required_cols %in% colnames(meta.all)], collapse = ", "))
}

# KO丰度
feat.ko <- read.csv(ko_file, header = TRUE, row.names = 1, check.names = FALSE)
if ("label" %in% colnames(feat.ko)) {
  feat.ko <- feat.ko[, !colnames(feat.ko) %in% "label", drop = FALSE]
}

# 物种丰度
feat.species <- read.csv(species_file, header = TRUE, row.names = 1, check.names = FALSE)
if ("label" %in% colnames(feat.species)) {
  feat.species <- feat.species[, !colnames(feat.species) %in% "label", drop = FALSE]
}

# 验证样本数和ID
cat("meta.all 样本数：", nrow(meta.all), "\n")
cat("feat.ko 样本数：", nrow(feat.ko), "\n")
cat("feat.species 样本数：", nrow(feat.species), "\n")

common_samples <- intersect(intersect(meta.all$sample_id, rownames(feat.ko)), rownames(feat.species))
if (length(common_samples) != 185) {
  cat("meta.all 中独有样本：", setdiff(meta.all$sample_id, common_samples), "\n")
  cat("feat.ko 中独有样本：", setdiff(rownames(feat.ko), common_samples), "\n")
  cat("feat.species 中独有样本：", setdiff(rownames(feat.species), common_samples), "\n")
  stop("三个表共有样本数仅为 ", length(common_samples), "，预期185")
}

# 对齐样本
meta.all <- meta.all[match(common_samples, meta.all$sample_id), ]
feat.ko <- feat.ko[common_samples, , drop = FALSE]
feat.species <- feat.species[common_samples, , drop = FALSE]

# 验证维度和样本ID
if (nrow(meta.all) != nrow(feat.ko) || nrow(meta.all) != nrow(feat.species)) {
  stop("对齐后维度不一致：meta.all=", nrow(meta.all), ", feat.ko=", nrow(feat.ko), ", feat.species=", nrow(feat.species))
}
if (!all(rownames(feat.ko) == meta.all$sample_id) || !all(rownames(feat.species) == meta.all$sample_id)) {
  stop("样本ID顺序不一致")
}

# 转置丰度表（MMUPHin要求行=特征，列=样本）
feat.ko <- t(feat.ko)
feat.species <- t(feat.species)

# 验证转置后样本匹配
if (!all(colnames(feat.ko) == rownames(meta.all))) {
  stop("feat.ko 列名与 meta.all 行名不匹配")
}
if (!all(colnames(feat.species) == rownames(meta.all))) {
  stop("feat.species 列名与 meta.all 行名不匹配")
}
cat("feat.ko 维度：", dim(feat.ko), "\n")
cat("feat.species 维度：", dim(feat.species), "\n")
cat("meta.all 维度：", dim(meta.all), "\n")

# 验证label与disease一致性
label_ko <- read.csv(ko_file, header = TRUE, row.names = 1, check.names = FALSE)$label
label_species <- read.csv(species_file, header = TRUE, row.names = 1, check.names = FALSE)$label
meta.all$label <- ifelse(meta.all$disease == "AD", 1, 0)
if (!all(label_ko[meta.all$sample_id] == meta.all$label, na.rm = TRUE) || !all(label_species[meta.all$sample_id] == meta.all$label, na.rm = TRUE)) {
  warning("KO或物种丰度表的label列与disease列不完全一致")
}

# 处理KO丰度数据
if (!all(sapply(feat.ko, is.numeric))) {
  stop("feat.ko 包含非数值列")
}
feat.ko[is.na(feat.ko)] <- 0
if (any(feat.ko < 0, na.rm = TRUE)) {
  warning("KO丰度发现负值，已替换为0")
  feat.ko[feat.ko < 0] <- 0
}
if (any(colSums(feat.ko, na.rm = TRUE) == 0)) {
  warning("KO丰度发现全零样本，添加伪计数1e-6")
  feat.ko <- feat.ko + 1e-6
}
feat.ko <- t(t(feat.ko) / colSums(feat.ko, na.rm = TRUE))  # 按样本归一化

# 处理物种丰度数据
if (!all(sapply(feat.species, is.numeric))) {
  stop("feat.species 包含非数值列")
}
feat.species[is.na(feat.species)] <- 0
if (any(feat.species < 0, na.rm = TRUE)) {
  warning("物种丰度发现负值，已替换为0")
  feat.species[feat.species < 0] <- 0
}
if (any(colSums(feat.species, na.rm = TRUE) == 0)) {
  warning("物种丰度发现全零样本，添加伪计数1e-6")
  feat.species <- feat.species + 1e-6
}
feat.species <- t(t(feat.species) / colSums(feat.species, na.rm = TRUE))  # 按样本归一化

# 3. 批次效应矫正
tryCatch({
  fit_ko_batch <- adjust_batch(
    feature_abd = feat.ko,
    batch = "HPC_Batch",
    covariates = c("Group", "Gender", "Age"),
    data = meta.all,
    control = list(verbose = TRUE)
  )
  ko_abd_adj <- fit_ko_batch$feature_abd_adj
}, error = function(e) {
  cat("KO批次矫正失败：", conditionMessage(e), "\n")
  stop("请检查feat.ko和meta.all的样本匹配")
})

tryCatch({
  fit_species_batch <- adjust_batch(
    feature_abd = feat.species,
    batch = "HPC_Batch",
    covariates = c("Group", "Gender", "Age"),
    data = meta.all,
    control = list(verbose = TRUE)
  )
  species_abd_adj <- fit_species_batch$feature_abd_adj
}, error = function(e) {
  cat("物种批次矫正失败：", conditionMessage(e), "\n")
  stop("请检查feat.species和meta.all的样本匹配")
})