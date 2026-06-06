import argparse
import yaml
from munch import Munch

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import models as models
from datasets.initialization import get_tta_dataset_corr
import pipeline.exp_geoid_corr as exp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default="configs/geoid/config_adapt_k2c.yaml",
                        type=str,
                        help="Path to config file")
    parser.add_argument('-ld', '--logdir', default='kitti_corr_adapt')
    parser.add_argument('-en', '--expname', default='ExpTTA_GeoID')
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    return args


def tta(cfg, args, corr_type, corr_level):
    test_dataset = get_tta_dataset_corr(cfg=cfg, debug=args.debug, corr_type=corr_type, corr_level=corr_level)
    
    if cfg.test_time_adaptation_params.online:
        cfg.test_time_adaptation_params.grad_steps = 1 
    
    test_loader_param = cfg.test_dataloader
    test_loader_param.pop('training')
    
    assert test_loader_param.batch_size == 1
    
    test_dataloader = DataLoader(test_dataset, **test_loader_param, collate_fn=test_dataset.collate_fn_tta_v2)
    
    model_ckpt_path = cfg.test_time_adaptation_params.model_ckpt_path
    
    ckpt_cfg = torch.load(model_ckpt_path, map_location='cuda')['hyper_parameters']['cfg']
    print(f'==> Test-time adaptation with ckpt {model_ckpt_path}')
    
    model = getattr(models, ckpt_cfg.model.name)(ckpt_cfg.model.in_feat_size, ckpt_cfg.model.out_classes)
    
    pl_module = getattr(exp, cfg.exp_name)(model, cfg)

    if args.debug:
        logger = TensorBoardLogger('logs_debug', name=cfg.log_dir, default_hp_metric=True)
    else:
        logger = TensorBoardLogger('logs_tta', name=cfg.log_dir, default_hp_metric=True)
    
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        )
    
    # Prepare test-time adaptation. Reset the target.
    pl_module.tta_setup(corr_type, corr_level)

    trainer = Trainer(max_epochs=1, gpus=1, 
                      logger=[logger],
                      callbacks=[checkpoint_callback])
    
    trainer.fit(pl_module, train_dataloaders=test_dataloader)


if __name__ == '__main__':
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    
    cfg.exp_name = args.expname
    cfg.log_dir = args.logdir

    if args.tta:
        corr_type_list = cfg.test_time_adaptation_params.corr_type_list
        corr_level_list = cfg.test_time_adaptation_params.corr_level_list
        for corr_type in corr_type_list:
            for corr_level in corr_level_list:
                cfg_ = cfg.copy()
                print(corr_type + '-' + corr_level)
                tta(cfg_, args, corr_type, corr_level)
