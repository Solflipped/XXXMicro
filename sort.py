import pandas as pd

# 1. 加载数据
df = pd.read_csv('/home/liang/project/XXXX-Transformer/Data/EW-T2D/species_abundance.csv')

# 2. 确定哪些列是“元数据”（不需要参与排序的列，如样本ID）
# 假设前两列是 sample_id 和 label，其余全是菌种名
metadata_columns = ['sample_id', 'label']
species_columns = [col for col in df.columns if col not in metadata_columns]

# 3. 对菌种名（特征名）进行字典顺序排序
# 排序后，所有 k__Bacteria... 会排在一起，其内部 p__Firmicutes... 又会排在一起
sorted_species_columns = sorted(species_columns)

# 4. 按照 [元数据列 + 排序后的特征列] 的顺序重新组合表格
# 这只会改变列的左右位置，不会改变单元格里的数值，也不会改变行的顺序
df_reordered = df[metadata_columns + sorted_species_columns]

# 5. 保存结果
df_reordered.to_csv('/home/liang/project/XXXX-Transformer/Data/EW-T2D/feature_sorted_abundance.csv', index=False)

print(f"成功！特征列已从 {len(species_columns)} 个调整为字典顺序。")