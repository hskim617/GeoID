GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_geoid.py \
--config configs/geoid/config_train_kitti_source.yaml \
--logdir kitti_source_train \
--expname ExpGeoID \
--savedir logs
