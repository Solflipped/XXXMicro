# source /home/liang/miniconda3/etc/profile.d/conda.sh  # PT5
# source /root/miniconda/etc/profile.d/conda.sh  # lanyunGPU
source /usr/local/miniconda3/etc/profile.d/conda.sh # ucloud
conda activate hj_env


# EW-T2D
# python main.py -d EW-T2D -f species -m FT_transformer --gpu 0 \
#   --n_blocks 1 \
#   --batch_size 8 \
#   --learning_rate 1e-3

# python main.py -d EW-T2_new -f ko -m KOFT --gpu 0 \
#   --d_token 96 \
#   --batch_size 16 \
#   --learning_rate 1e-4

python main.py -d EW-T2_new -f species -m UFEN --gpu 0 \
  --d_token 64 \
  --batch_size 16 \
  --learning_rate 1e-4

# python main.py -d EW-T2D -f species,ko -m MBT --gpu 0 \
#   --n_blocks 2 \
#   --fusion_layer 2 \
#   --num_bottleneck 4 \
#   --batch_size 8 \
#   --learning_rate 1e-4 


# python main.py -d EW-T2D -f ko,species -m MSFT --gpu 0 \
#   --n_layers 2 \
#   --batch_size 32 \
#   --num_bottleneck 4 \
#   --use_bottleneck --use_cross_atn --btn_init "embed" \
#   --learning_rate 1e-4

# python main.py -d EW-T2D -f ko,species -m MDL4Microbiome --gpu 0 \
#   --batch_size 8 \
#   --learning_rate 1e-4


# python main.py -d EW-T2D -f ko,species -m XXXMicro --gpu 0 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --btn_init embed \
#   --dropout 0.1 \
#   --d_token 32 \
#   --n_query 8 \
#   --num_bottleneck 4 \
#   --n_attn_heads 4 \
#   --n_enhance_layers 1 \
#   --n_fusion_layers 2 \
#   --mask_ratio 0.01 \
#   --lambda_recon 0.05


# C-T2D
# python main.py -d C-T2D -f species -m FT_transformer --gpu 0 \
#   --n_blocks 2 \
#   --batch_size 16 \
#   --learning_rate 1e-5



# LC
# python main.py -d LC -f species -m FT_transformer --gpu 0 \
#   --n_blocks 2 \
#   --batch_size 8 \
#   --learning_rate 1e-4

# python main.py -d LC -f ko,species -m MSFTTransformer --gpu 0 \
#   --n_layers 1 \
#   --batch_size 8 \
#   --num_bottleneck 2 \
#   --use_bottleneck --use_cross_atn --btn_init "embed" \
#   --learning_rate 0.001

# python main.py -d LC -f species -m UFEN --gpu 0 \
#   --d_token 64 \
#   --batch_size 16 \
#   --learning_rate 1e-4
