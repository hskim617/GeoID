GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_geoid_corr.py \
--config configs/geoid/config_adapt_n2c.yaml \
--logdir nusc_corr_adapt \
--expname ExpTTA_GeoID \
--tta
