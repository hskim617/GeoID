from stat import FILE_ATTRIBUTE_ENCRYPTED
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import yaml
import os

from utils.utils import get_semcolor_common, get_semcolor_kitti, get_semcolor_nusc
from utils.ply_vis import write_ply
from utils.eval import SemanticEval
import utils.losses as losses

import MinkowskiEngine as ME
import copy
from tqdm import tqdm

import models as models


class ExpBase(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        
        self.save_hyperparameters(ignore='model')
        
        self.model = model
        
        self.model_params = cfg.model
        self.train_params = cfg.train_params
        self.generalization_params = cfg.generalization_params
        
        self.exp_params = cfg.exp_params
        self.log_freq = cfg.exp_params.log_freq
        
        if self.train_params.use_pretrained:
            self.load_pretrained(self.model, self.train_params.pretrained_ckpt_path, 'model')
            print(f'==> Load pretrained weights from {self.train_params.pretrained_ckpt_path}')
        
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def load_pretrained(self, model, dict_path, model_name):
        state_dict = torch.load(dict_path)['state_dict']
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if model_name in key:
                new_state_dict[key.replace(model_name + '.', '')] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(new_state_dict, strict=True)
        
        
class ExpSemantic(ExpBase):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        
        ds = cfg.dataset_SemKITTI.voxel_size
        self.voxel_size = ds
        self.quantization = torch.Tensor([ds, ds, ds])
        self.class_num = cfg.model.out_classes
        self.common_map = yaml.safe_load(open('configs/label_mapping/common_map_merged.yaml', 'r'))

        self.class_str_common = self.common_map['labels']
        self.source = cfg.generalization_params.source
        self.target = cfg.generalization_params.target
    
        self.target_mapping = {}
        for i, target in enumerate(self.target):
            self.target_mapping[i] = target
        
        if self.source == 'kitti':
            self.source_map = yaml.safe_load(open('configs/label_mapping/semantic-kitti.yaml', 'r'))
            learning_map_inv = self.source_map['learning_map_inv']
            self.class_str = {}
            for k, v in learning_map_inv.items():
                self.class_str[k] = self.source_map['labels'][v]
        elif self.source == 'nuscenes':
            self.source_map = yaml.safe_load(open('configs/label_mapping/nuscenes.yaml', 'r'))
            self.class_str = self.source_map['labels_16']
                    
        self.configure_evaluator()
        self.configure_criterion()
        
    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.model.parameters()),
                         lr=self.train_params.learning_rate)
        return optimizer
        
    def configure_evaluator(self):
        self.evaluators = []
        
        target = copy.deepcopy(self.target)
        for t in target:
            if self.source in t:
                self.evaluators.append(SemanticEval(len(self.class_str), None, [0]))
            else:
                self.evaluators.append(SemanticEval(len(self.class_str_common), None, [0]))
        
        for evaluator in self.evaluators:
            evaluator.reset()
        return
    
    def configure_criterion(self):
        if self.train_params.criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        elif self.train_params.criterion == 'dice':
            self.criterion = losses.DICELoss(ignore_label=-1)
            
        elif self.train_params.criterion == 'softdice':
            self.criterion = losses.SoftDICELoss(ignore_label=-1)
            
    def test_setup(self, target, visualization, log_freq=50):
        self.target = target
        self.target_mapping = {}
        for i, target in enumerate(self.target):
            self.target_mapping[i] = target
        
        self.configure_evaluator()
        self.visualization = visualization
        self.test_log_freq = log_freq

        return
    
    def forward(self, batch):
        coordinates = batch['coordinates'] # N, 3
        features = batch['features'] # N, 1
        
        sparse_tensor = ME.SparseTensor(coordinates=coordinates.int(), features=features)
        predicted_sparse_tensor, _ = self.model(sparse_tensor)
        out = predicted_sparse_tensor
            
        return out
    
    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        
        out = self.forward(batch)
        preds = out.F

        loss = 0
        semantic_vox_loss =  self.criterion(preds, labels-1)
        loss += semantic_vox_loss
        
        return {'loss': loss}
    
    def training_epoch_end(self, training_step_outputs):
        
        loss = 0
        for output in training_step_outputs:
            loss += output['loss']
        
        self.log(f'train/loss', loss / len(training_step_outputs), rank_zero_only=True)
        return
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch['batch_size']
        
        with torch.no_grad():
            out = self.forward(batch)

            for b_idx in range(batch_size):
                feats = out.features_at(b_idx)[batch['inverse_maps'][b_idx]]
                predictions = torch.add(torch.argmax(feats, dim=1), 1).cpu().numpy()
                labels = batch['labels'][b_idx].cpu().numpy()
                
                
                if self.target_mapping[dataloader_idx] == self.source:
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)
                else:
                    # common mapping
                    if self.source == 'kitti':
                        class_remap_source2common = self.common_map['learning_map_from_kitti_to_common']
                        if self.target_mapping[dataloader_idx] == 'nuscenes':
                            class_remap_target2common = self.common_map['learning_map_from_nusc_to_common']
                        else:
                            raise NotImplementedError
                    elif self.source == 'nuscenes':
                        class_remap_source2common = self.common_map['learning_map_from_nusc_to_common']
                        if self.target_mapping[dataloader_idx] == 'kitti':
                            class_remap_target2common = self.common_map['learning_map_from_kitti_to_common']
                        else:
                            raise NotImplementedError
                    maxkey_source2common = max(class_remap_source2common.keys())
                    maxkey_target2common = max(class_remap_target2common.keys())
                    remap_lut_source2common = np.zeros((maxkey_source2common + 100), dtype=np.int32)
                    remap_lut_source2common[list(class_remap_source2common.keys())] = list(class_remap_source2common.values())
                    remap_lut_target2common = np.zeros((maxkey_target2common + 100), dtype=np.int32)
                    remap_lut_target2common[list(class_remap_target2common.keys())] = list(class_remap_target2common.values())
                    
                    predictions = remap_lut_source2common[predictions]
                    labels = remap_lut_target2common[labels]
                    
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)
        return
    
    def validation_epoch_end(self, val_step_outputs):
        if self.global_step == 0: return
        
        output_list = []
        average = 0
        num = 0
        for idx, evaluator in enumerate(self.evaluators):
            iou, output_dict = self.log_metric(evaluator, self.target_mapping[idx])
            output_list.append(output_dict)
            
            self.log(f'valid/mIoU/{self.target_mapping[idx]}', iou, rank_zero_only=True)
            self.log(f'{self.target_mapping[idx]}/mIoU/all', iou, rank_zero_only=True)
            
            for k, v in output_dict.items():
                if k in ['all', 'unlabeled', 'noise']:
                    continue
                self.log(f'{self.target_mapping[idx]}/IoU/{k}', v['IoU'], rank_zero_only=True)
            
            evaluator.reset()
            average += iou
            num += 1
        
        assert num > 0, "need at least one validation set"
        
        self.log(f'valid/mIoU/average', average/num, rank_zero_only=True)
        self.write_txt(output_list, average=average/num)
        return
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch['batch_size']
        
        with torch.no_grad():
            out = self.forward(batch)

            for b_idx in range(batch_size):
                feats = out.features_at(b_idx)[batch['inverse_maps'][b_idx]]
                predictions = torch.add(torch.argmax(feats, dim=1), 1).cpu().numpy()
                labels = batch['labels'][b_idx].cpu().numpy()
                
                if self.source in self.target_mapping[dataloader_idx]:
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)
                else:
                    # common mapping
                    if self.source == 'kitti':
                        class_remap_source2common = self.common_map['learning_map_from_kitti_to_common']
                        if self.target_mapping[dataloader_idx] == 'nuscenes':
                            class_remap_target2common = self.common_map['learning_map_from_nusc_to_common']
                        else:
                            raise NotImplementedError
                    elif self.source == 'nuscenes':
                        class_remap_source2common = self.common_map['learning_map_from_nusc_to_common']
                        if self.target_mapping[dataloader_idx] == 'kitti':
                            class_remap_target2common = self.common_map['learning_map_from_kitti_to_common']
                        else:
                            raise NotImplementedError
                    maxkey_source2common = max(class_remap_source2common.keys())
                    maxkey_target2common = max(class_remap_target2common.keys())
                    remap_lut_source2common = np.zeros((maxkey_source2common + 100), dtype=np.int32)
                    remap_lut_source2common[list(class_remap_source2common.keys())] = list(class_remap_source2common.values())
                    remap_lut_target2common = np.zeros((maxkey_target2common + 100), dtype=np.int32)
                    remap_lut_target2common[list(class_remap_target2common.keys())] = list(class_remap_target2common.values())
                    
                    predictions = remap_lut_source2common[predictions]
                    labels = remap_lut_target2common[labels]
                    
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)

                # log
                if batch['batch_idxs'][b_idx].item() % self.test_log_freq == 0:
                    iou, output_dict = self.log_metric(self.evaluators[dataloader_idx], self.target_mapping[dataloader_idx])
                    self.log(f'test/running_mIoU/{self.target_mapping[dataloader_idx]}', iou, on_step=True, rank_zero_only=True)
                    for k, v in output_dict.items():
                        if k in ['all', 'unlabeled', 'noise']:
                            continue
                        self.log(f'test_{self.target_mapping[dataloader_idx]}/running_IoU/{k}', v['IoU'], on_step=True, rank_zero_only=True)
                
                # visualize
                if self.visualization:
                    save_dir = os.path.join(self.logger.log_dir, self.target_mapping[dataloader_idx])
                    os.makedirs(save_dir, exist_ok=True)
                    save_filename = os.path.join(save_dir, f"{batch_idx:05}")
                    
                    bg_mask = labels == 0
                    predictions[bg_mask] = 0
                    
                    coords = out.coordinates_at(b_idx)
                    self.visualize(save_filename, coords.cpu().numpy(), predictions, labels, self.target_mapping[dataloader_idx])
        return
    
    def test_epoch_end(self, test_step_outputs):
        output_list = []
        average = 0
        num = 0
        for idx, evaluator in enumerate(self.evaluators):
            iou, output_dict = self.log_metric(evaluator, self.target_mapping[idx])
            output_list.append(output_dict)

            self.log(f'test/mIoU/{self.target_mapping[idx]}', iou, rank_zero_only=True)
            self.log(f'{self.target_mapping[idx]}/mIoU/all', iou, rank_zero_only=True)
            
            for k, v in output_dict.items():
                if k in ['all', 'unlabeled', 'noise']:
                    continue
                self.log(f'{self.target_mapping[idx]}/IoU/{k}', v['IoU'], rank_zero_only=True) 
            
            evaluator.reset()
            average += iou
            num += 1
        
        assert num > 0, "need at least one test set"
        
        self.log(f'test/mIoU/average', average/num, rank_zero_only=True)
        self.write_txt(output_list, average=average/num)
        return
    
    def add_prediction(self, target, evaluator, predictions, labels):
        assert isinstance(evaluator, SemanticEval)
        
        if target == self.source:
            evaluator.addBatchSemIoU(predictions, labels)
        else:
            # common mapping
            if self.source == 'kitti':
                if target == 'nuscenes':
                    class_remap_source2common = self.common_map['learning_map_from_kitti_to_common']
                    class_remap_target2common = self.common_map['learning_map_from_nusc_to_common']
                elif target == 'poss':
                    class_remap_source2common = self.common_map['learning_map_from_kitti_to_common_sksp']
                    class_remap_target2common = self.common_map['learning_map_from_poss_to_common_sksp']
                else:
                    raise NotImplementedError
            elif self.source == 'nuscenes':
                if target == 'kitti':
                    class_remap_source2common = self.common_map['learning_map_from_nusc_to_common']
                    class_remap_target2common = self.common_map['learning_map_from_kitti_to_common']
                elif target == 'poss':
                    class_remap_source2common = self.common_map['learning_map_from_nusc_to_common_nssp']
                    class_remap_target2common = self.common_map['learning_map_from_poss_to_common_nssp']
                else:
                    raise NotImplementedError
            maxkey_source2common = max(class_remap_source2common.keys())
            maxkey_target2common = max(class_remap_target2common.keys())
            remap_lut_source2common = np.zeros((maxkey_source2common + 100), dtype=np.int32)
            remap_lut_source2common[list(class_remap_source2common.keys())] = list(class_remap_source2common.values())
            remap_lut_target2common = np.zeros((maxkey_target2common + 100), dtype=np.int32)
            remap_lut_target2common[list(class_remap_target2common.keys())] = list(class_remap_target2common.values())
            
            predictions = remap_lut_source2common[predictions]
            labels = remap_lut_target2common[labels]
            
            evaluator.addBatchSemIoU(predictions, labels)
        
    def visualize(self, filename, coords, preds, labels, target):
        if target == self.source:
            if self.source == 'kitti':
                color_preds = get_semcolor_kitti(preds)
                color_labels = get_semcolor_kitti(labels)
            elif self.source == 'nuscenes':
                color_preds = get_semcolor_nusc(preds)
                color_labels = get_semcolor_nusc(labels)
            else:
                raise NotImplementedError
        else:
            color_preds = get_semcolor_common(preds)
            color_labels = get_semcolor_common(labels)
        
        write_ply(filename + '-gt.ply', [coords, color_labels], ['x','y','z','red','green','blue'])
        write_ply(filename + '-pd.ply', [coords, color_preds], ['x','y','z','red','green','blue'])
        return
    
    def log_metric(self, evaluator, target):
        assert isinstance(evaluator, SemanticEval)

        if self.source in target:
            class_str = self.class_str
        else:
            class_str = self.class_str_common
        
        class_IoU, class_all_IoU = evaluator.getSemIoU()

        # now make a nice dictionary
        output_dict = {}

        # make python variables
        class_IoU = class_IoU.item()
        class_all_IoU = class_all_IoU.flatten().tolist()

        output_dict["all"] = {}
        output_dict["all"]["IoU"] = class_IoU

        for idx, iou in enumerate(class_all_IoU):
            c_str = class_str[idx]
            output_dict[c_str] = {}
            output_dict[c_str]["IoU"] = iou

        mIoU = output_dict["all"]["IoU"]
        return mIoU, output_dict
    
    def write_txt(self, output_dict, average=None):
        file_dir = os.path.join(self.logger.log_dir, 'metric.txt')
        fd = open(file_dir, 'a')
        fd.write('-'*(144+99) + '\n')
        fd.write('|'+ ' '*60 + f'Epoch {self.current_epoch:02}' + ' '*(3) + f'Step {self.global_step:06}' + ' '*(60+99) + '|\n')
                
        for idx, output in enumerate(output_dict):
            fd.write('|{}'.format('target'.ljust(10)))
            for k in output.keys():
                fd.write('|{}'.format(k.ljust(10)[:10]))
            fd.write('|\n')
            
            # fd.write('|{}'.format(' '*10))
            fd.write('|{}'.format(self.target_mapping[idx].ljust(10)))
            for v in output.values():
                fd.write('|{:.4f}    '.format(v['IoU']))
            fd.write('|\n')
        
        if average is not None:
            fd.write('|{}'.format('average'.ljust(10)))
            fd.write('|{:.4f}    '.format(average))
            fd.write('|\n')
        
        fd.write('-'*(144+99) + '\n')
        
        fd.close()
        return


