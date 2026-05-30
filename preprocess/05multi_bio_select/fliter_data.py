import numpy as np
import pandas as pd


INFO_COLS = ["sample_id", "label"]


def _to_relative_abundance(feature_data: pd.DataFrame, input_kind: str) -> pd.DataFrame:
    """将特征矩阵转换为相对丰度（0~1）"""
    features = feature_data.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)

    if input_kind == "percentage":
        # 输入是 0~100 百分比
        rel_abund = features / 100.0
    elif input_kind == "proportion":
        # 输入已经是 0~1 相对丰度
        rel_abund = features.copy()
    elif input_kind == "count":
        # 输入是 count，按样本总量归一化
        row_sums = features.sum(axis=1)
        rel_abund = features.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    else:
        raise ValueError(f"未知 input_kind: {input_kind}")

    return rel_abund


def filter_abundance_data(
    file_path: str,
    output_path: str,
    input_kind: str,
    global_min_rel_abund: float = 5e-4,
    prevalence_min_rel_abund: float = 1e-6,
    prevalence_ratio: float = 0.10,
    enable_rule2: bool = True,
) -> pd.DataFrame:
    """
    过滤规则：
    1) 保留至少在一个样本中相对丰度 >= global_min_rel_abund 的特征
    2) (可选) 保留在至少 prevalence_ratio 样本中相对丰度 >= prevalence_min_rel_abund 的特征
    """
    df = pd.read_csv(file_path)
    if not set(INFO_COLS).issubset(df.columns):
        raise ValueError(f"{file_path} 缺少必要列: {INFO_COLS}")

    info_df = df[INFO_COLS].copy()
    feature_data = df.drop(columns=INFO_COLS).copy()
    rel_abund = _to_relative_abundance(feature_data, input_kind=input_kind)

    # 标准1：全体样本中至少有1个样本达到 0.05%（可配置）
    mask1 = (rel_abund >= global_min_rel_abund).any(axis=0)
    n_samples = rel_abund.shape[0]

    if enable_rule2:
        min_samples = max(1, int(np.ceil(prevalence_ratio * n_samples)))
        mask2 = (rel_abund >= prevalence_min_rel_abund).sum(axis=0) >= min_samples
        final_mask = mask1 & mask2
    else:
        min_samples = None
        final_mask = mask1

    filtered_features = feature_data.loc[:, final_mask]
    result_df = pd.concat([info_df, filtered_features], axis=1)

    print("=" * 80)
    print(f"输入文件: {file_path}")
    print(f"input_kind: {input_kind}")
    print(f"样本数: {n_samples}")
    print(f"原始特征数: {feature_data.shape[1]}")
    print(f"标准1阈值: 相对丰度 >= {global_min_rel_abund}")
    if enable_rule2:
        print(
            f"标准2阈值: 相对丰度 >= {prevalence_min_rel_abund} 且检出样本数 >= {min_samples}/{n_samples}"
        )
    else:
        print("标准2阈值: 已关闭")
    print(f"过滤后特征数: {filtered_features.shape[1]}")
    print(f"输出文件: {output_path}")

    result_df.to_csv(output_path, index=False)
    return result_df


def run_default_filters() -> None:
    """按当前项目路径批量处理 species / KO / AMP """
    base_dir = "/root/XXXMicro/Data/AD"
    
    amp_use_10pct_prevalence = False

    configs = [
        # species: 阈值 1e-4 & 10% samples
        {
            "name": "species",
            "file_path": f"{base_dir}/preprocess_species_abundance.csv",
            "output_path": f"{base_dir}/merged_species.csv",
            "input_kind": "percentage",
            "global_min_rel_abund": 5e-4,
            "prevalence_min_rel_abund": 1e-6,
            "prevalence_ratio": 0.10,
            "enable_rule2": True,
        },
        # KO: 阈值 1e-6 & 10% samples
        # {
        #     "name": "ko",
        #     "file_path": f"{base_dir}/preprocess_ko_abundance.csv",
        #     "output_path": f"{base_dir}/merged_ko.csv",
        #     "input_kind": "count",
        #     "global_min_rel_abund": 5e-4,
        #     "prevalence_min_rel_abund": 1e-6,
        #     "prevalence_ratio": 0.10,
        #     "enable_rule2": True,
        # },
        # AMP: 默认 1e-6 & 5% samples，可切换到
        {
            "name": "amp",
            "file_path": f"{base_dir}/preprocess_amp_abundance.csv",
            "output_path": f"{base_dir}/merged_amp.csv",
            "input_kind": "count",
            "global_min_rel_abund": 5e-4,
            "prevalence_min_rel_abund": 1e-6,
            "prevalence_ratio": 0.10 if amp_use_10pct_prevalence else 0.05,
            "enable_rule2": True,
        },
    ]

    for cfg in configs:
        print(f"\n开始处理: {cfg['name']}")
        filter_abundance_data(
            file_path=cfg["file_path"],
            output_path=cfg["output_path"],
            input_kind=cfg["input_kind"],
            global_min_rel_abund=cfg["global_min_rel_abund"],
            prevalence_min_rel_abund=cfg["prevalence_min_rel_abund"],
            prevalence_ratio=cfg["prevalence_ratio"],
            enable_rule2=cfg["enable_rule2"],
        )


if __name__ == "__main__":
    run_default_filters()
