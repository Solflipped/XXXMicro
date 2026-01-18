source /home/liang/miniconda3/etc/profile.d/conda.sh
conda activate hj_env
python test_pytorch.py


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

python main.py -d EW-T2D -f species -m FTMicro --gpu 0 \
  --batch_size 8 \
  --num_conv_layers 2 \
  --d_token 128 \
  --learning_rate 1e-4 


