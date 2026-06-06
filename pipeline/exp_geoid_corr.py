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

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

import models as models
from pipeline.exp import ExpBase


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
            
    def test_setup(self, target, visualization, log_freq=50, corr_type_list=None, corr_level_list=None):
        self.target = target
        self.corr_type_list = corr_type_list
        self.corr_level_list = corr_level_list
        
        self.target_mapping = []
        for i in corr_type_list:
            for j in corr_level_list:
                self.target_mapping.append(target[0]+"_"+i+"_"+j)
        
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

        if 'LIDAR_TOP' in batch['scan_ids'][0]: # nusc
            corr_type = batch['scan_ids'][0].split('/')[-5]
            corr_level = batch['scan_ids'][0].split('/')[-4]
        else: # kitti
            corr_type = batch['scan_ids'][0].split('/')[-4]
            corr_level = batch['scan_ids'][0].split('/')[-3]
        
        self.corr_type = corr_type
        self.corr_level = corr_level
        
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
                    
                    self.log(f'test/running_mIoU/{self.target_mapping[dataloader_idx]}/{corr_type}_{corr_level}', iou, on_step=True, rank_zero_only=True)
                    for k, v in output_dict.items():
                        if k in ['all', 'unlabeled', 'noise']:
                            continue
                        self.log(f'test_{self.target_mapping[dataloader_idx]}/{corr_type}_{corr_level}/running_IoU/{k}', v['IoU'], on_step=True, rank_zero_only=True)
                
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
            
            self.log(f'test/mIoU/{self.target_mapping[idx]}/{self.corr_type}_{self.corr_level}', iou, rank_zero_only=True)
            self.log(f'{self.target_mapping[idx]}/{self.corr_type}_{self.corr_level}/mIoU/all', iou, rank_zero_only=True)
            
            for k, v in output_dict.items():
                if k in ['all', 'unlabeled', 'noise']:
                    continue
                self.log(f'{self.target_mapping[idx]}/{self.corr_type}_{self.corr_level}/IoU/{k}', v['IoU'], rank_zero_only=True) 
            
            evaluator.reset()
            average += iou
            num += 1
        
        assert num > 0, "need at least one test set"
        
        self.log(f'test/mIoU/average', average/num, rank_zero_only=True)
        self.write_txt_corr(output_list, average=average/num, 
                            corr_type_list=self.corr_type_list,
                            corr_level_list=self.corr_level_list)
        return
    
    def add_prediction(self, target, evaluator, predictions, labels):
        assert isinstance(evaluator, SemanticEval)
        
        if self.source in target:
            evaluator.addBatchSemIoU(predictions, labels)
        else:
            # common mapping
            if self.source == 'kitti':
                class_remap_source2common = self.common_map['learning_map_from_kitti_to_common']
                if target == 'nuscenes':
                    class_remap_target2common = self.common_map['learning_map_from_nusc_to_common']
                else:
                    raise NotImplementedError
            elif self.source == 'nuscenes':
                class_remap_source2common = self.common_map['learning_map_from_nusc_to_common']
                if target == 'kitti':
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
            
            evaluator.addBatchSemIoU(predictions, labels)
        
    def visualize(self, filename, coords, preds, labels, target):
        
        if self.source in target:
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
    
    def log_confusion(self, evaluator, target):
        assert isinstance(evaluator, SemanticEval)
        confusion = evaluator.getConfMatrix()
        
        # row: true label, column: predicted label
        fig, ax = plot_confusion_matrix(conf_mat=confusion.T, figsize=(9,9), colorbar=True, show_absolute=False, show_normed=True, class_names=list(self.class_str.values()))
        plt.savefig(f'{self.logger.log_dir}/conf_{target}.png', dpi=900)
        plt.clf()

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

    def write_txt_corr(self, output_dict, average=None, corr_type_list=None, corr_level_list=None):
        len_corr_type = len(corr_type_list)
        len_corr_level = len(corr_level_list)
        print(f'len_output: {len(output_dict)}, num_corr: {len_corr_type*len_corr_level}\n')
        file_dir = os.path.join(self.logger.log_dir, 'metric.txt')
        fd = open(file_dir, 'a')
        fd.write('-'*(144+99) + '\n')
        fd.write(self.target_mapping[0].split('_')[0]+"_c" + "\n")
        
        corr_type_level_list = []
        for corr_type in corr_type_list:
            for corr_level in corr_level_list:
                corr_type_level_list.append([corr_type, corr_level])
        
        for idx, output in enumerate(output_dict):
            if idx == 0:
                fd.write('|{}'.format('corr_type'.ljust(10)))
                fd.write('|{}'.format('corr_level'.ljust(10)))
                for k in output.keys():
                    fd.write('|{}'.format(k.ljust(10)[:10]))
                fd.write('|\n')
            fd.write('|{}'.format(corr_type_level_list[idx][0].ljust(10)[:10]))
            fd.write('|{}'.format(corr_type_level_list[idx][1].ljust(10)[:10]))
            for v in output.values():
                fd.write('|{:.4f}    '.format(v['IoU']))
            fd.write('|\n')

        if average is not None:
            fd.write('|{}'.format('average'.ljust(10)))
            fd.write('|{}'.format('average'.ljust(10)))
            fd.write('|{:.4f}    '.format(average))
            fd.write('|\n')
        
        fd.write('-'*(144+99) + '\n')
        
        fd.close()
        return
    
    def write_txt_corr_single(self, output_dict, average=None, corr_type=None, corr_level=None):
        file_dir = os.path.join(self.logger.log_dir, 'metric' + corr_type + '_' + corr_level + '.txt')
        fd = open(file_dir, 'a')
        fd.write('-'*(144+99) + '\n')
        fd.write('|'+ ' '*60 + f'Epoch {self.current_epoch:02}' + ' '*(3) + f'Step {self.global_step:06}' + ' '*(60+99) + '|\n')
        
        for idx, output in enumerate(output_dict):
            fd.write('|{}'.format('target'.ljust(10)))
            for k in output.keys():
                fd.write('|{}'.format(k.ljust(10)[:10]))
            fd.write('|\n')
            
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



#! TTA baseline
class ExpTTA_Base(ExpSemantic):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        
        # TTA parameters
        self.tta_params = cfg.test_time_adaptation_params
        
        self.test_time_adaptation = self.tta_params.test_time_adaptation
        if self.test_time_adaptation:
            self.automatic_optimization = False

    ### TTA stuff ###
    def tta_setup(self, corr_type, corr_level):
        # reset the target
        self.target = [self.tta_params.target]
        
        self.target_mapping = {}
        for i, target in enumerate(self.target):
            self.target_mapping[i] = target
        
        self.corr_type = corr_type
        self.corr_level = corr_level
        
        # reset evaluators
        self.configure_evaluator()

        if self.tta_params.online:
            # load model
            # build optimizer
            self.load_pretrained(self.model, self.tta_params.model_ckpt_path, 'model')
            self.tta_optimizer = optim.Adam(list(self.model.parameters()),
                                            lr=self.tta_params.learning_rate_tta,
                                            weight_decay=self.tta_params.weight_decay_tta)
            self.tta_params.grad_steps = 1
            
        if self.tta_params.grad_steps == 0:
            self.load_pretrained(self.model, self.tta_params.model_ckpt_path, 'model')
        
        self.tta_log_freq = self.tta_params.log_freq
        return
    
    def set_model_train(self):
        self.model.train()
        
        if self.tta_params.disable_bn_adaptation:  # disable statistical alignment
            for m in self.model.modules():
                if isinstance(m, ME.MinkowskiBatchNorm):
                    m.eval()
        return
    
    def model_device(self, model):
        for p in model.parameters():
            return p.device
        for b in model.buffers():
            return b.device
        return None


from torch_cluster import knn
from pipeline.exp_geoid import ExpGeoID
class ExpTTA_GeoID(ExpTTA_Base, ExpGeoID):
    '''
    TTA with geometric inlier discrimination (GeoID) task.
    '''
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        
        self.tta_geoid_params = cfg.tta_geoid_params
        self.criterion_cls = nn.BCEWithLogitsLoss()
        
        ####! define teacher model for geoid !####
        self.apply_noise_mask = self.tta_params.get('apply_noise_mask', True)
        self.bidirection_mask = self.tta_params.get('bidirection_mask', True)
        if self.apply_noise_mask:
            self.teacher_model = getattr(models, cfg.model.name)(cfg.model.in_feat_size, cfg.model.out_classes)
            self.load_pretrained(self.teacher_model, self.tta_params.model_ckpt_path, 'model')
                
        self.noise_threshold = self.tta_params.get('noise_threshold', 0.5)
        self.threshold_margin_orig = self.tta_params.get('threshold_margin_orig', 0.0)
        self.threshold_margin_noise = self.tta_params.get('threshold_margin_noise', 0.0)
        
    def freeze_model(self, model, key):
        for name, param in model.named_parameters():
            if key in name:
                param.requires_grad = False
        return
    
    def check_gradient(self):
        max_grad = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_max = param.grad.abs().max().item()
                if param_max > max_grad:
                    max_grad = param_max
        return max_grad
        
    def training_step(self, batch, batch_idx):
        if self.test_time_adaptation:
            return self.tta_step(batch, batch_idx)
        else:
            pass
        return
        
    def training_epoch_end(self, training_step_outputs):
        if self.test_time_adaptation:
            return self.tta_epoch_end(training_step_outputs)
        else:
            pass
        return

    def forward_tta(self, coordinates, features, is_cls=False):
        
        sparse_tensor = ME.SparseTensor(coordinates=coordinates.int(), features=features)
        
        if is_cls:
            out_cls = self.model(sparse_tensor, is_cls=True)
            return out_cls
        
        out_seg, out_cls = self.model(sparse_tensor)
        return out_seg, out_cls
    
    def forward_tta_teacher(self, coordinates, features, is_cls=False):
        
        sparse_tensor = ME.SparseTensor(coordinates=coordinates.int(), features=features)
        
        if is_cls:
            out_cls = self.teacher_model(sparse_tensor, is_cls=True)
            return out_cls
        
        out_seg, out_cls = self.teacher_model(sparse_tensor)
        return out_seg, out_cls
    
    def tta_step(self, batch, batch_idx):
        
        # labels for evaluation
        labels = batch['labels_orig']
        inverse_maps = batch['inverse_maps_orig']
        
        batch_size = batch['batch_size']
        assert batch_size == 1, 'tta batch size should be 1'
        
        # test-only loop
        if self.tta_params.test_only:
            assert self.tta_params.grad_steps == 0, 'grad step should be 0 for test only option'
            
            self.model.eval()
            with torch.no_grad():
                out_seg, out_cls = self.forward_tta(batch['coordinates_orig'], batch['features_orig'])
                
                pt_predictions = torch.add(torch.argmax(out_seg.F, dim=1), 1)
                pt_predictions = pt_predictions[inverse_maps].cpu().numpy()
                pt_labels = labels.cpu().numpy()
            
            self.add_prediction(self.target_mapping[0], self.evaluators[0], pt_predictions, pt_labels)
            
            # logging
            if (batch_idx + 1) % self.tta_log_freq == 0:
                iou, output_dict = self.log_metric(self.evaluators[0], self.target_mapping[0])
                self.logger.experiment.add_scalar(f'test/running_mIoU/{self.target_mapping[0]}/{self.corr_type}_{self.corr_level}', iou, batch_idx+1)
                
                for k, v in output_dict.items():
                    if k in ['all', 'unlabeled', 'noise']:
                        continue
                    self.logger.experiment.add_scalar(f'test_{self.target_mapping[0]}_{self.corr_type}_{self.corr_level}/running_IoU/{k}', v['IoU'], batch_idx+1)
            return

        # prepare TTA
        if not self.tta_params.online:
            self.load_pretrained(self.model, self.tta_params.model_ckpt_path, 'model')
            self.tta_optimizer = optim.Adam(list(self.model.parameters()),
                                                lr=self.tta_params.learning_rate_tta,
                                                weight_decay=self.tta_params.weight_decay_tta)
    
        # Input for geoid
        with torch.no_grad():
            # make batch
            batch_coordinates = batch['coordinates_orig'].repeat(self.tta_params.inner_batch_size_tta, 1)
            batch_features = batch['features_orig'].repeat(self.tta_params.inner_batch_size_tta, 1)
            batch_id = torch.arange(self.tta_params.inner_batch_size_tta, device=self.device).unsqueeze(1).repeat(1, batch['coordinates_orig'].shape[0]).flatten()
            batch_coordinates[:, 0] += batch_id
            
            # define noise
            valid_idx = batch_features.shape[0]
            coord_orig = batch_coordinates
            
            if self.tta_geoid_params.geoid_type == 'frame':
                while True:
                    batch_select = torch.where(torch.rand(self.tta_params.inner_batch_size_tta) > self.tta_geoid_params.geoid_p)[0]
                    if len(batch_select) != 0: break
                
                selected_idx = (coord_orig[:,0].unsqueeze(1) == batch_select.to(self.device)).any(dim=1)
                selected_idx = torch.where(selected_idx)[0]
                selected_coordinates = coord_orig[selected_idx]
                noise_coords = self.noise_generator_random_configure(selected_coordinates,
                                                                     min_dist=self.tta_geoid_params.min_dist,
                                                                     max_dist=self.tta_geoid_params.max_dist, 
                                                                     geoid_configure_type=self.tta_geoid_params.geoid_configure_type)
                
            elif self.tta_geoid_params.geoid_type == 'point':
                while True:
                    if isinstance(self.tta_geoid_params.geoid_p, list):
                        point_p = np.random.uniform(self.tta_geoid_params.geoid_p[0], self.tta_geoid_params.geoid_p[1])
                        rand_select = torch.where(torch.rand(coord_orig.shape[0]) > point_p)[0]
                    else:
                        rand_select = torch.where(torch.rand(coord_orig.shape[0]) > self.tta_geoid_params.geoid_p)[0]
                    if len(rand_select) > 0: break
                selected_coordinates = coord_orig[rand_select]
                noise_coords = self.noise_generator_random_configure(selected_coordinates,
                                                                     min_dist=self.tta_geoid_params.min_dist,
                                                                     max_dist=self.tta_geoid_params.max_dist,
                                                                     geoid_configure_type=self.tta_geoid_params.geoid_configure_type)
            
            assert noise_coords is not None
            
            # filter unique coordinates
            noise_coords = torch.unique(noise_coords, dim=0)
            
            # delete overlap noise
            merged_coords = copy.deepcopy(torch.cat([coord_orig, noise_coords], 0))
            q_coords, q_select, q_inverse = ME.utils.sparse_quantize(merged_coords, quantization_size=1, return_index=True, return_inverse=True)
            
            unique_coords = torch.ones_like(q_select)
            overlap = q_inverse[:valid_idx]
            unique_coords[overlap] = 0
            
            uniuqe_noise = unique_coords[q_inverse]
            uniuqe_noise = torch.where(uniuqe_noise==1)[0]
            
            # final coordinates
            noise_coords = merged_coords[uniuqe_noise]
            merged_coords = torch.cat([coord_orig, noise_coords], dim=0)
            
            index = knn(coord_orig[:,1:], noise_coords[:,1:], k=1, batch_x=coord_orig[:,0].long(), batch_y=noise_coords[:,0].long())
            noise_features = batch_features[index[1]]
                
                
            merged_features = torch.cat([batch_features, noise_features], dim=0)
            
            # make labels for geoid
            pruning_mask = torch.zeros(merged_coords.shape[0], device=self.device)
            pruning_mask[:valid_idx] = 1
            pruning_mask = pruning_mask > 0.5
            labels_cls = pruning_mask.float()
            
            
            # bidirectional unreliable point filtering
            if self.apply_noise_mask:
                self.teacher_model.eval()
                
                out_cls = self.forward_tta_teacher(coord_orig, batch_features, is_cls=True)
                loss_mask_valid = (out_cls.F.squeeze(1).sigmoid() > (self.noise_threshold - self.threshold_margin_orig))
                loss_mask = torch.ones(merged_coords.shape[0], dtype=bool, device=self.device)
                loss_mask[:valid_idx] = loss_mask_valid

                if self.bidirection_mask:
                    out_cls = self.forward_tta_teacher(merged_coords, merged_features, is_cls=True)
                    loss_mask_noise = (out_cls.F.squeeze(1).sigmoid() < (self.noise_threshold + self.threshold_margin_noise))
                    loss_mask[valid_idx:] = loss_mask_noise[valid_idx:]
            
        self.model.zero_grad()
        self.set_model_train()
        
        # TTA loop
        for grad_step in range(self.tta_params.grad_steps):
            
            # model forward
            out_cls = self.forward_tta(merged_coords, merged_features, is_cls=True)
            
            preds_cls = out_cls.F
                        
            # geoid loss
            if self.apply_noise_mask:
                cls_loss = self.criterion_cls(preds_cls.squeeze(1)[loss_mask], labels_cls[loss_mask])
            else:
                cls_loss = self.criterion_cls(preds_cls.squeeze(1), labels_cls)
            
            cls_loss = self.tta_params.get('loss_weight', 1.0) * cls_loss


            # compute losses
            loss = torch.tensor(0.0, device=self.device)
            loss += cls_loss

            # logging initial loss
            if grad_step == 0:
                self.logger.experiment.add_scalar(f'test_loss/{self.target_mapping[0]}/{self.corr_type}_{self.corr_level}/init_loss', loss.item(), batch_idx+1)
            
            # optimize
            self.tta_optimizer.zero_grad()
            loss.backward()

            # clip gradients
            if self.tta_params.get('clip_gradient', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.tta_optimizer.step()
            self.model.zero_grad()

        # logging final loss
        self.logger.experiment.add_scalar(f'test_loss/{self.target_mapping[0]}/{self.corr_type}_{self.corr_level}', loss.item(), batch_idx+1)

        # inference
        self.model.eval()
                        
        with torch.no_grad():
            out_seg, out_cls = self.forward_tta(batch['coordinates_orig'], batch['features_orig'])
            pt_predictions = torch.add(torch.argmax(out_seg.F, dim=1), 1)[inverse_maps]
            pt_predictions = pt_predictions.cpu().numpy()
            pt_labels = labels.cpu().numpy()
            
        self.add_prediction(self.target_mapping[0], self.evaluators[0], pt_predictions, pt_labels)
        
        if self.tta_params.visualize:
            if batch_idx % 5 == 0:
                save_dir = f'{self.logger.log_dir}/visualize'
                os.makedirs(save_dir, exist_ok=True)
                filename = f'{save_dir}/{batch_idx}-tta'

                pt_coords = batch['xyz_orig'].detach().cpu().numpy()
                self.visualize(filename, pt_coords, pt_predictions, pt_labels, self.target_mapping[0])
            
        # logging
        if (batch_idx + 1) % self.tta_log_freq == 0:
            iou, output_dict = self.log_metric(self.evaluators[0], self.target_mapping[0])
            self.logger.experiment.add_scalar(f'test/running_mIoU/{self.target_mapping[0]}/{self.corr_type}_{self.corr_level}', iou, batch_idx+1)
            
            for k, v in output_dict.items():
                if k in ['all', 'unlabeled', 'noise']:
                    continue
                self.logger.experiment.add_scalar(f'test_{self.target_mapping[0]}_{self.corr_type}_{self.corr_level}/running_IoU/{k}', v['IoU'], batch_idx+1)    
        return
    
    def tta_epoch_end(self, training_step_outputs):
        iou, output_dict = self.log_metric(self.evaluators[0], self.target_mapping[0])
        print(f'iou: {iou}\n')
        
        final_iter = len(self.trainer.train_dataloader.dataset.datasets)
        
        self.logger.experiment.add_scalar(f'test/running_mIoU/{self.target_mapping[0]}/{self.corr_type}_{self.corr_level}', iou, final_iter)
        for k, v in output_dict.items():
            if k in ['all', 'unlabeled', 'noise']:
                continue
            self.logger.experiment.add_scalar(f'test_{self.target_mapping[0]}_{self.corr_type}_{self.corr_level}/running_IoU/{k}', v['IoU'], final_iter)
        
        self.log(f'test_final/mIoU/{self.target_mapping[0]}/{self.corr_type}_{self.corr_level}', iou, rank_zero_only=True)
        self.write_txt_corr_single([output_dict], average=iou, corr_type=self.corr_type, corr_level=self.corr_level)
        
        self.log_confusion(self.evaluators[0], self.target_mapping[0])
        
        return


