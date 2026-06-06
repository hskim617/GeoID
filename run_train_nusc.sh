GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_geoid.py \
--config configs/geoid/config_train_nusc_source.yaml \
--logdir nuscenes_source_train \
--expname ExpGeoID \
--savedir logs


# 11/11
# CUDA_VISIBLE_DEVICES=3 python main_noise_corr.py -c configs/noise/run_knn/n2c_v3_tta_p10_d1-3_bs2_lr0p001_upfv3_margin0p1.yaml -ld n2c_v3_knn_tta_p10_d1-3_bs2_lr0p001_upfv3_margin0p1 -en ExpTTA_noise --tta
# CUDA_VISIBLE_DEVICES=3 python main_noise_corr.py -c configs/noise/run_knn/n2c_v3_tta_p10_d1-3_bs2_lr0p001_upfv3_margin0p4.yaml -ld n2c_v3_knn_tta_p10_d1-3_bs2_lr0p001_upfv3_margin0p4 -en ExpTTA_noise --tta
