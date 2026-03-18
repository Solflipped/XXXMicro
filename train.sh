# source /home/liang/miniconda3/etc/profile.d/conda.sh  # PT5
source /root/miniconda/etc/profile.d/conda.sh  # lanyunGPU
conda activate hj_env


# EW-T2D
 python main.py -d EW-T2D -f species -m FT_transformer --gpu 0 \
  --n_blocks 4 \
  --batch_size 8 \
  --learning_rate 1e-4 

# python main.py -d EW-T2D -f ko -m UFEN --gpu 0 \
#   --d_token 64 \
#   --base_channels 64 \
#   --expansion_factor 2 \
#   --num_layers 4 \
#   --latent_dim 512 \
#   --batch_size 8 \
#   --learning_rate 1e-4 

# python main.py -d EW-T2D -f species,ko -m MBT --gpu 0 \
#   --n_blocks 4 \
#   --fusion_layer 2 \
#   --num_bottleneck 4 \
#   --batch_size 8 \
#   --learning_rate 1e-4 

# python main.py -d EW-T2D -f species -m FT_transformer --gpu 0 \
#   --batch_size 8 \
#   --n_blocks 4 \
#   --learning_rate 1e-4 

# python main.py -d EW-T2D -f ko,species -m FTMicro --gpu 0 \
#   --batch_size 16 \
#   --d_token 96 \
#   --fusion_depth 4 \
#   --dst_embedding_length 4 \
#   --ahl_depth 3 \
#   --learning_rate 1e-4

  
# python main.py -d EW-T2D -f ko,species -m MSFTTransformer --gpu 0 \
#   --n_blocks 2 \
#   --batch_size 16 \
#   --num_bottleneck 4 \
#   --learning_rate 1e-4
