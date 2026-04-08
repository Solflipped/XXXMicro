import os
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from train import train
from utils import check_record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Models on custom dataset.')
    # GPU ID
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    # Disease
    parser.add_argument('-d','--disease', type=str, default='AD', help='Disease type')
    # Feature type
    parser.add_argument('-f', '--feature', type=str, default='ko,species', help='Feature type (e.g., ko or species)')
    # model type
    parser.add_argument('-m', '--model_type', type=str, default='UFEN', help='model type (UFEN)')
    # Batch size
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for training')
    # Learning rate
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    # cvfold
    parser.add_argument('-c','--cvfold', type=int, default=5, help="The value of k in k-fold cross validation.  (default: 5)")
    
    # model params
    # UFEN
    parser.add_argument('--d_token', type=int, default=64, help='Token channel size for UFEN')
    args = parser.parse_args()

    if args.model_type == "UFEN":
        if ',' in args.feature:
            raise ValueError("UFEN only supports unimodal features (e.g., 'ko' or 'species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'd_token': args.d_token,
        }
    else:
        assert False, f"{args.model_type} type not supported"

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Training configuration:", args)

    seeds = [42, 1024, 2023, 8888, 12345] # 设定 5 个用于重复验证的种子
    results_dir = os.path.join('./results', args.disease)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f'{args.model_type}.csv')

    # 将超参数打包成 OrderedDict，作为实验的“唯一身份识别码”
    record = OrderedDict({
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "feature": args.feature,
        **{k: v for k, v in params.items() if k not in ['lr', 'batch_size']}
    })

    if not check_record(record, log_path):
        print("该超参数组合已训练过，跳过本次任务。")
        exit(0)

    all_seed_scores = [] # 用于收集每个 seed 跑出来的平均结果
    seed_records = []

    for seed in seeds:
        seed_scores = train(
            disease=args.disease,
            feature=args.feature,
            model_type=args.model_type,
            cvfold=args.cvfold,
            seed=seed,
            **params
        )

        if seed_scores:
            all_seed_scores.append(seed_scores)

            seed_record = OrderedDict({
                "seed": seed,
                **record,
                **seed_scores,
            })
            seed_records.append(seed_record)

    # 全局汇总与结果保存
    if all_seed_scores:
        print("\n========== Final Results ==========")
        
        summary_record = OrderedDict({"seed": "all"})
        summary_record.update(record)

        for key in all_seed_scores[0].keys():
            vals = [s[key] for s in all_seed_scores]
            mean = np.mean(vals)
            std = np.std(vals)
            summary_record[key] = f"{mean:.4f} ± {std:.4f}"
            print(f"{key}: {mean:.4f} ± {std:.4f}")

        out_df = pd.DataFrame(seed_records + [summary_record])

        if os.path.exists(log_path):
            old_df = pd.read_csv(log_path)
            out_df = pd.concat([old_df, out_df], ignore_index=True)

        out_df.to_csv(log_path, index=False)
        print(f"\n 实验全部完成，结果及全局统计已成功追加至 {log_path}")