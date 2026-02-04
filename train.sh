source /home/liang/miniconda3/etc/profile.d/conda.sh
conda activate hj_env


# EW-T2D
# python main.py -d EW-T2D -f ko,species -m MBT --gpu 0 \
#   --n_blocks 4 \
#   --fusion_layer 2 \
#   --num_bottleneck 4 \
#   --mbt_use_cross_atn \
#   --batch_size 8 \
#   --learning_rate 1e-4 

# python main.py -d EW-T2D -f species -m FT_transformer --gpu 0 \
#   --batch_size 8 \
#   --n_blocks 4 \
#   --learning_rate 1e-4 

python main.py -d EW-T2D -f species,ko -m FTMicro --gpu 0 \
  --batch_size 8 \
  --d_token 96 \
  --num_layers 4 \
  --base_channels 96 \
  --expansion_factor 2 \
  --latent_dim 512 \
  --fusion_depth 2 \
  --dst_embedding_length 8 \
  --ahl_depth 3 \
  --learning_rate 1e-4