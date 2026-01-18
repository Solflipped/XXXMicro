import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Models on custom dataset.')
    # GPU ID
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    # Disease
    parser.add_argument('-d','--disease', type=str, default='AD', help='Disease type')
    # Feature type
    parser.add_argument('-f', '--feature', type=str, default='ko,species', help='Feature type (e.g., ko or species)')
    # model type
    parser.add_argument('-m', '--model_type', type=str, default='FT_transformer', help='model type (FT_transformer, FTMicro, MBT, MDL4Microbiome, MSFTTransformer, FT_Vote, GDFT, GAFT)')
    # Batch size
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for training')
    # Learning rate
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')

    # model params
    # MBT 
    parser.add_argument('-fl', '--fusion_layer', type=int, default=2, help='MBT fusion layer')
    parser.add_argument('-num_b', '--num_bottleneck', type=int, default=4, help='Number of bottleneck (MBT & MSFTTransformer)')

    # FT_transformer / FT_Vote / MSFTTransformer / MBT
    parser.add_argument('-nb', '--n_blocks', type=int, default=4, help='Number of transformer blocks')

    # FTMicro params
    parser.add_argument('-ncl', '--num_conv_layers', type=int, default=2, help='Number of convolutional layers (FTMicro)')
    parser.add_argument('-dt', '--d_token', type=int, default=192, help='Token embedding dimension (FTMicro)')

    # GAFT params
    parser.add_argument('--lmf_hidden_dim', type=int, default=128, help='LMF SubNet hidden dim (per modality)')
    parser.add_argument('--lmf_output_dim', type=int, default=128, help='LMF fused output dim')
    parser.add_argument('--lmf_rank', type=int, default=4, help='LMF rank for low-rank fusion')
    parser.add_argument('--lmf_dropout', type=float, default=0.1, help='Dropout in LMF subnet and post-fusion')
    parser.add_argument('--use_lmf_subnet', action='store_true', help='Use SubNet pre-processing in LMF')
    parser.add_argument('--gat_dim', type=int, default=128, help='Hidden dim inside GAT (GAFT)')
    parser.add_argument('--gat_dropout', type=float, default=0.1, help='Dropout used in GAT attention/projection (GAFT)')
    parser.add_argument('--finetune_mbt', action='store_true', help='If set, finetune MBT inside GAFT (default freeze)')
    # MBT cross-attention alignment toggle
    parser.add_argument('--mbt_use_cross_atn', action='store_true', help='Enable pre-encoder cross-attention alignment inside MBT')

    args = parser.parse_args()

    # FT_transformer params
    if args.model_type == "FT_transformer":
        params = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'n_blocks': args.n_blocks
        }
    elif args.model_type == "FTMicro":
        params = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_conv_layers': args.num_conv_layers,
            'd_token': args.d_token,
        }
    elif args.model_type == "MBT":
        # MBT必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("MBT requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'n_blocks': args.n_blocks,
            'fusion_layer': args.fusion_layer,
            'num_bottleneck': args.num_bottleneck,
            'mbt_use_cross_atn': args.mbt_use_cross_atn,
        }
    elif args.model_type == "MDL4Microbiome":
        # MDL4Microbiome 必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("MDL4Microbiome requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
        }
    elif args.model_type == "MSFTTransformer":
        # MSFTTransformer 必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("MSFTTransformer requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'n_blocks': args.n_blocks,
            'num_bottleneck': args.num_bottleneck,
        }
    elif args.model_type == "FT_Vote":
        # FT_Vote 必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("FT_Vote requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'n_blocks': args.n_blocks,
        }
    elif args.model_type == "GAFT":
        # GAFT 必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("GAFT requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'n_blocks': args.n_blocks,
            'fusion_layer': args.fusion_layer,
            'num_bottleneck': args.num_bottleneck,
            'lmf_hidden_dim': args.lmf_hidden_dim,
            'lmf_output_dim': args.lmf_output_dim,
            'lmf_rank': args.lmf_rank,
            'lmf_dropout': args.lmf_dropout,
            'use_lmf_subnet': args.use_lmf_subnet,
            'gat_dim': args.gat_dim,
            'gat_dropout': args.gat_dropout,
            'finetune_mbt': args.finetune_mbt,
            'mbt_use_cross_atn': args.mbt_use_cross_atn,
        }
    
    else:
        assert False, f"{args.model_type} type not supported"

    # Set GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Training configuration:", args)

    train(
        disease=args.disease,
        feature=args.feature,
        model_type=args.model_type,
        **params
    )
