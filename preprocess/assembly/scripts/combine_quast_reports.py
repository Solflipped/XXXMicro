import pandas as pd # 注意提前安装好pandas
import os

sample_names = snakemake.params['sample_names']
assembly_dir = snakemake.params['assembly_dir']

merge_df = pd.DataFrame()
for sample in sample_names:
    print(f"deal with sample: {sample}")
    infile = os.path.join(assembly_dir, sample, 'quast/report.tsv')
    if not os.path.exists(infile):
        print(f"********** Cant find {infile}! ***********")
        continue
    quast_df = pd.read_table(infile, header=0, index_col=0)
    merge_df = pd.concat([merge_df, quast_df], axis=1)

# 清理列名：去掉 .contig 后缀
merge_df.columns = merge_df.columns.str.replace(r'\.contig$', '', regex=True)

# 转置，让样本做行
merge_df = merge_df.T

# 输出结果
merge_df.to_csv(snakemake.output[0], sep='\t', index=True, index_label="Sample")
