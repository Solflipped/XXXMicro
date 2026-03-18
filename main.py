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
    parser.add_argument('-m', '--model_type', type=str, default='FT_transformer', help='model type (FT_transformer, UFEN, FTMicro, MBT, MDL4Microbiome, MSFTTransformer, FT_Vote)')
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

    # FTMicro、UFEN params
    parser.add_argument('-dt', '--d_token', type=int, default=96, help='Token embedding dimension (FTMicro、UFEN)')
    parser.add_argument('-fd', '--fusion_depth', type=int, default=4, help='Fusion depth (FTMicro)')
    parser.add_argument('-del', '--dst_embedding_length', type=int, default=4, help='Destination embedding length (FTMicro)')
    parser.add_argument('-ad', '--ahl_depth', type=int, default=3, help='AHL depth (FTMicro)')
    parser.add_argument('-bc', '--base_channels', type=int, default=96, help='Base channels (UFEN)')
    parser.add_argument('-ef', '--expansion_factor', type=int, default=2, help='Channel expansion factor (UFEN)')
    parser.add_argument('-nl', '--num_layers', type=int, default=4, help='Encoder/decoder layers (UFEN)')
    parser.add_argument('-ldm', '--latent_dim', type=int, default=512, help='Latent dimension (UFEN)')
    


    # MBT cross-attention alignment toggle
    parser.add_argument('--use_cross_atn', action='store_true', help='Enable pre-encoder cross-attention alignment inside MBT')

    args = parser.parse_args()


    # FT_transformer params
    if args.model_type == "FT_transformer":
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'n_blocks': args.n_blocks
        }
    elif args.model_type == "UFEN":
        # UFEN 只支持单模态特征
        if ',' in args.feature:
            raise ValueError("UFEN only supports unimodal features (e.g., 'ko' or 'species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'd_token': args.d_token,
            'base_channels': args.base_channels,
            'expansion_factor': args.expansion_factor,
            'num_layers': args.num_layers,
            'latent_dim': args.latent_dim,
        }
    elif args.model_type == "FTMicro":
        # FTMicro 必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("FTMicro requires multimodal features (e.g., 'ko,species' or 'species,ko').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'd_token': args.d_token,
            'fusion_depth': args.fusion_depth,
            'dst_embedding_length': args.dst_embedding_length,
            'ahl_depth': args.ahl_depth,
        }
    elif args.model_type == "MBT":
        # MBT必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("MBT requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'n_blocks': args.n_blocks,
            'fusion_layer': args.fusion_layer,
            'num_bottleneck': args.num_bottleneck,
            'use_cross_atn': args.use_cross_atn,
        }
    elif args.model_type == "MDL4Microbiome":
        # MDL4Microbiome 必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("MDL4Microbiome requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
        }
    elif args.model_type == "MSFTTransformer":
        # MSFTTransformer 必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("MSFTTransformer requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'n_blocks': args.n_blocks,
            'num_bottleneck': args.num_bottleneck,
        }
    elif args.model_type == "FT_Vote":
        # FT_Vote 必须使用多模态特征
        if ',' not in args.feature:
            raise ValueError("FT_Vote requires multimodal features (e.g., 'ko,species').")
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'n_blocks': args.n_blocks,
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
        seeds=[392, 412, 432, 452, 472], 
        **params
    )
