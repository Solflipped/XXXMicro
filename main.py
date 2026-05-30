import os
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
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
    parser.add_argument('-m', '--model_type', type=str, default='UFEN', help='model type (UFEN \\ KOFT \\ XXXMicro)')
    # Batch size
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for training')
    # Learning rate
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    # cvfold
    parser.add_argument('-c','--cvfold', type=int, default=5, help="The value of k in k-fold cross validation.  (default: 5)")

    # model params
    # UFEN / KOFT / XXXMicro
    parser.add_argument('--d_token', type=int, default=64, help='Token channel size for UFEN / KOFT / XXXMicro')
    # XXXMicro
    parser.add_argument('--n_query', type=int, default=4, help='Number of query tokens for XXXMicro')
    parser.add_argument('--num_bottleneck', type=int, default=4, help='Number of bottleneck tokens for XXXMicro')
    parser.add_argument('--n_enhance_layers', type=int, default=2, help='Number of enhance layers for XXXMicro')
    parser.add_argument('--n_fusion_layers', type=int, default=2, help='Number of fusion layers for XXXMicro')
    parser.add_argument('--n_attn_heads', type=int, default=4, help='Number of attention heads for XXXMicro')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout for XXXMicro')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Mask ratio for XXXMicro')
    parser.add_argument('--btn_init', type=str, default='embed', choices=['embed', 'random'], help='Bottleneck initialization strategy for XXXMicro')
    parser.add_argument('--lambda_recon', type=float, default=0.5, help='Weight for reconstruction loss in XXXMicro')
    args = parser.parse_args()

    if args.model_type == "UFEN":
        if ',' in args.feature:
            raise ValueError("UFEN only supports unimodal features (e.g., 'ko' or 'species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'd_token': args.d_token,
        }
    elif args.model_type == "KOFT":
        if ',' in args.feature:
            raise ValueError("KOFT only supports unimodal features (e.g., 'ko' or 'species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'd_token': args.d_token,
        }
    elif args.model_type == "XXXMicro":
        if ',' not in args.feature:
            raise ValueError("XXXMicro requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'd_token': args.d_token,
            'n_query': args.n_query,
            'num_bottleneck': args.num_bottleneck,
            'n_enhance_layers': args.n_enhance_layers,
            'n_fusion_layers': args.n_fusion_layers,
            'n_attn_heads': args.n_attn_heads,
            'dropout': args.dropout,
            'mask_ratio': args.mask_ratio,
            'btn_init': args.btn_init,
            'lambda_recon': args.lambda_recon,
        }
    else:
        assert False, f"Model type '{args.model_type}' is not supported!"

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Import after setting CUDA_VISIBLE_DEVICES, otherwise torch may initialize
    # CUDA before the env var takes effect.
    from train import train

    print("Training configuration:", args)
    # seeds = [42, 777, 1024] # 设定 3 个用于重复验证的种子
    # seeds = [392, 412, 432, 452, 472]
    seeds = [392, 412, 432]
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

    check_dict = OrderedDict({"seed": "all"})
    check_dict.update(record) # 构造check_dict用于检查是否已存在完整实验记录（包含all汇总）

    if not check_record(check_dict, log_path):
        print("该超参数组合的完整实验(已包含all汇总)已训练过，跳过本次任务。")
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
