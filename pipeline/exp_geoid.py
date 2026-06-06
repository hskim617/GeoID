from stat import FILE_ATTRIBUTE_ENCRYPTED
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import yaml
import os
import random
from utils.eval import SemanticEval

import MinkowskiEngine as ME
import copy

import models as models
from pipeline.exp import ExpSemantic

# from knn_cuda import KNN
from torch_cluster import knn


class ExpGeoID_Base(ExpSemantic):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        
        self.geoid_params = cfg.geoid_params
        self.criterion_cls = nn.BCEWithLogitsLoss()
    
    def configure_evaluator(self):
        self.evaluators = []
        self.evaluators_noised = []
        
        target = copy.deepcopy(self.target)
        for t in target:
            if self.source in t:
                self.evaluators.append(SemanticEval(len(self.class_str), None, [0]))
                self.evaluators_noised.append(SemanticEval(len(self.class_str), None, [0]))
            else:
                self.evaluators.append(SemanticEval(len(self.class_str_common), None, [0]))
                self.evaluators_noised.append(SemanticEval(len(self.class_str_common), None, [0]))
        
        for evaluator in self.evaluators:
            evaluator.reset()
        
        for evaluator in self.evaluators_noised:
            evaluator.reset()
        return
           
    def noise_generator_random_configure(self, coord_orig, min_dist=1, max_dist=1, geoid_configure_type='axis'):
        
        if geoid_configure_type == 'axis':
            pad_ = [[0,1,0,0], [0,-1,0,0], [0,0,1,0], [0,0,-1,0], [0,0,0,1], [0,0,0,-1]]
            pad_ = torch.Tensor(pad_)
            
            pad_ori = torch.randint(0, 6, (coord_orig.shape[0],))
            pad_range = torch.randint(min_dist, max_dist+1, (coord_orig.shape[0],1))
            
            pad_T = pad_[pad_ori]
            pad_T = pad_T * pad_range
            
        else:
            raise NotImplementedError
        
        noise_coord = torch.add(coord_orig, pad_T.to(self.device))
        
        return noise_coord
    
    def forward(self, batch):
        
        coordinates = batch['coordinates'] # N, 3
        features = batch['features'] # N, 1
        
        valid_idx = features.shape[0]
        
        if self.geoid_params.geoid_type == 'frame':
            batch_select = torch.where(torch.rand(batch['batch_size']) > self.geoid_params.geoid_p)[0]
            if len(batch_select) != 0:
                batch_idx = coordinates[:,0]
                selected_idx = (batch_idx.unsqueeze(1) == batch_select.to(self.device)).any(dim=1)
                selected_idx = torch.where(selected_idx)[0]
                selected_coordinates = coordinates[selected_idx]
                noise_coords = self.noise_generator_random_configure(selected_coordinates, min_dist=self.geoid_params.min_dist, max_dist=self.geoid_params.max_dist, geoid_configure_type=self.geoid_params.geoid_configure_type)
            else:
                noise_coords = None
                
        elif self.geoid_params.geoid_type == 'point':
            if isinstance(self.geoid_params.geoid_p, list):
                point_p = np.random.uniform(self.geoid_params.geoid_p[0], self.geoid_params.geoid_p[1])
                rand_select = torch.where(torch.rand(coordinates.shape[0]) > point_p)[0]
            else:
                rand_select = torch.where(torch.rand(coordinates.shape[0]) > self.geoid_params.geoid_p)[0]
            
            if len(rand_select) != 0:
                selected_coordinates = coordinates[rand_select]
                noise_coords = self.noise_generator_random_configure(selected_coordinates, min_dist=self.geoid_params.min_dist, max_dist=self.geoid_params.max_dist, geoid_configure_type=self.geoid_params.geoid_configure_type)
            else:
                noise_coords = None
        
        else:
            raise NotImplementedError

    
        if noise_coords is not None:
            # filter unique coordinates
            noise_coords = torch.unique(noise_coords, dim=0)
            
            merged_coords = copy.deepcopy(torch.cat([coordinates, noise_coords], 0))
            q_coords, q_select, q_inverse = ME.utils.sparse_quantize(merged_coords, quantization_size=1, return_index=True, return_inverse=True)
            
            unique_coords = torch.ones_like(q_select)
            
            overlap = q_inverse[:valid_idx]
            
            unique_coords[overlap] = 0
            
            uniuqe_noise = unique_coords[q_inverse]
            uniuqe_noise = torch.where(uniuqe_noise==1)[0]
            
            
            noise_coords = merged_coords[uniuqe_noise]

            index = knn(coordinates[:,1:], noise_coords[:,1:], k=1, batch_x=coordinates[:,0].long(), batch_y=noise_coords[:,0].long())
            noise_feats = features[index[1]]

        if self.training:
            if noise_coords is not None:
                coordinates = torch.cat([coordinates, noise_coords], dim=0)
                features = torch.cat([features, noise_feats], dim=0)
                
            sparse_tensor = ME.SparseTensor(coordinates=coordinates.int(), features=features)
            out_seg, out_cls = self.model(sparse_tensor)
            
            pruning_mask = torch.zeros(out_seg.F.shape[0], device=self.device)
            pruning_mask[:valid_idx] = 1
            pruning_mask = pruning_mask > 0.5
            pruning = ME.MinkowskiPruning()
            out_seg = pruning(out_seg, pruning_mask)
            
            labels_cls = pruning_mask.float()
            
            return out_seg, out_cls, labels_cls
            
        else:
            # extend original data
            assert batch['batch_size'] == 1
            
            if noise_coords is not None:
                coordinates_noised = torch.cat([coordinates, noise_coords], dim=0)
                features_noised = torch.cat([features, noise_feats], dim=0)
            else:
                coordinates_noised = copy.deepcopy(coordinates)
                features_noised = copy.deepcopy(features)
            
            coordinates_noised[:,0] += batch['batch_size']
            
            coordinates = torch.cat((coordinates, coordinates_noised), dim=0)
            features = torch.cat((features, features_noised), dim=0)
            
            sparse_tensor = ME.SparseTensor(coordinates=coordinates.int(), features=features)
            out_seg, out_cls = self.model(sparse_tensor)

            pruning_mask = torch.zeros(out_seg.F.shape[0], device=self.device)
            pruning_mask[:valid_idx] = 1 # orig
            pruning_mask[:valid_idx + valid_idx] = 1 # orig of noised
            pruning_mask = pruning_mask > 0.5
            pruning = ME.MinkowskiPruning()
            out_seg = pruning(out_seg, pruning_mask.to(self.device))
            
            labels_cls = pruning_mask.float()
        
        return out_seg, out_cls, labels_cls
    
    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        
        out_seg, out_cls, labels_cls = self.forward(batch)
        preds = out_seg.F
        preds_cls = out_cls.F

        loss = 0
        seg_loss = self.criterion(preds, labels-1)
        loss += seg_loss
        
        cls_loss = self.criterion_cls(preds_cls.squeeze(1), labels_cls.float())
        loss += cls_loss
        
        self.log('seg', seg_loss.item(), prog_bar=True, logger=False, on_epoch=False)
        self.log('cls', cls_loss.item(), prog_bar=True, logger=False, on_epoch=False)
        
        return {'loss': loss, 'seg_loss': seg_loss.item(), 'cls_loss': cls_loss.item()}
    
    def training_epoch_end(self, training_step_outputs):
        
        loss = 0
        seg_loss = 0
        cls_loss = 0
        for output in training_step_outputs:
            loss += output['loss']
            seg_loss += output['seg_loss']
            cls_loss += output['cls_loss']
        
        self.log(f'train/loss', loss / len(training_step_outputs), rank_zero_only=True)
        self.log(f'train/seg_loss', seg_loss / len(training_step_outputs), rank_zero_only=True)
        self.log(f'train/cls_loss', cls_loss / len(training_step_outputs), rank_zero_only=True)
        return
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch['batch_size']
        assert batch_size == 1
        
        with torch.no_grad():
            out_seg, out_cls, labels_cls = self.forward(batch)

            for b_idx in range(batch_size):
                feats = out_seg.features_at(b_idx)[batch['inverse_maps'][b_idx]]
                predictions = torch.add(torch.argmax(feats, dim=1), 1).cpu().numpy()
                labels = batch['labels'][b_idx].cpu().numpy()
                
                # seg prediction for noised point cloud
                feats_noised = out_seg.features_at(b_idx+1)[batch['inverse_maps'][b_idx]]
                predictions_noised = torch.add(torch.argmax(feats_noised, dim=1), 1).cpu().numpy()
                
                if self.target_mapping[dataloader_idx] == self.source:
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)
                    self.evaluators_noised[dataloader_idx].addBatchSemIoU(predictions_noised, labels)
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
                    
                    predictions_noised = remap_lut_source2common[predictions_noised]
                    
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)
                    self.evaluators_noised[dataloader_idx].addBatchSemIoU(predictions_noised, labels)
                
                # geoid prediction
                predictions_cls_all = out_cls.F.sigmoid().squeeze(1) > 0.5                
                acc_all = (predictions_cls_all == labels_cls).float().mean()
                
                predictions_cls = out_cls.features_at(b_idx).sigmoid().squeeze(1) > 0.5
                acc = (predictions_cls == labels_cls[out_cls.coordinates[:,0] == b_idx]).float().mean()
                
                predictions_cls_noised = out_cls.features_at(b_idx+1).sigmoid().squeeze(1) > 0.5
                acc_noised = (predictions_cls_noised == labels_cls[out_cls.coordinates[:,0] == (b_idx+1)]).float().mean()
                
        return {'acc_all': acc_all.item(), 'acc': acc.item(), 'acc_noised': acc_noised.item()}
    
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
        
        output_list = []
        average = 0
        num = 0
        # noised evaluation
        for idx, evaluator in enumerate(self.evaluators_noised):
            iou, output_dict = self.log_metric(evaluator, self.target_mapping[idx])
            output_list.append(output_dict)
            
            self.log(f'valid/mIoU/{self.target_mapping[idx]}_noised', iou, rank_zero_only=True)
            self.log(f'{self.target_mapping[idx]}_noised/mIoU/all', iou, rank_zero_only=True)
            
            for k, v in output_dict.items():
                if k in ['all', 'unlabeled', 'noise']:
                    continue
                self.log(f'{self.target_mapping[idx]}_noised/IoU/{k}', v['IoU'], rank_zero_only=True)
            
            evaluator.reset()
            average += iou
            num += 1
        
        assert num > 0, "need at least one validation set"
        
        self.log(f'valid/mIoU/average_noised', average/num, rank_zero_only=True)
        self.write_txt(output_list, average=average/num, noised=True)
        
        # geoid prediction
        if len(self.target) == 1:
            val_step_outputs = [val_step_outputs]
        for i, val_step_output in enumerate(val_step_outputs):
            acc_all = 0
            acc = 0
            acc_noised = 0
            for output in val_step_output:
                acc_all += output['acc_all']
                acc += output['acc']
                acc_noised += output['acc_noised']
            self.log(f'valid/geoid_acc_all/{self.target_mapping[i]}', acc_all / len(val_step_output), rank_zero_only=True)
            self.log(f'valid/geoid_acc/{self.target_mapping[i]}', acc / len(val_step_output), rank_zero_only=True)
            self.log(f'valid/geoid_acc_noised/{self.target_mapping[i]}', acc_noised / len(val_step_output), rank_zero_only=True)
        
        return
    
    def write_txt(self, output_dict, average=None, noised=False):
        file_dir = os.path.join(self.logger.log_dir, 'metric.txt')
        fd = open(file_dir, 'a')
        fd.write('-'*(144+99) + '\n')
        if noised:
            fd.write('|'+ ' '*60 + f'Epoch Noised {self.current_epoch:02}' + ' '*(3) + f'Step {self.global_step:06}' + ' '*(60+99-9) + '|\n')
        else:
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


class ExpGeoID(ExpGeoID_Base):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        
    def forward(self, batch):
        
        coordinates = batch['coordinates'] # N, 3
        features = batch['features'] # N, 1
        
        valid_idx = features.shape[0]
        
        if self.geoid_params.geoid_type == 'frame':
            batch_select = torch.where(torch.rand(batch['batch_size']) > self.geoid_params.geoid_p)[0]
            if len(batch_select) != 0:
                batch_idx = coordinates[:,0]
                selected_idx = (batch_idx.unsqueeze(1) == batch_select.to(self.device)).any(dim=1)
                selected_idx = torch.where(selected_idx)[0]
                selected_coordinates = coordinates[selected_idx]
                noise_coords = self.noise_generator_random_configure(selected_coordinates, min_dist=self.geoid_params.min_dist, max_dist=self.geoid_params.max_dist, geoid_configure_type=self.geoid_params.geoid_configure_type)
            else:
                noise_coords = None
                
        elif self.geoid_params.geoid_type == 'point':
            if isinstance(self.geoid_params.geoid_p, list):
                point_p = np.random.uniform(self.geoid_params.geoid_p[0], self.geoid_params.geoid_p[1])
                rand_select = torch.where(torch.rand(coordinates.shape[0]) > point_p)[0]
            else:
                rand_select = torch.where(torch.rand(coordinates.shape[0]) > self.geoid_params.geoid_p)[0]
            
            if len(rand_select) != 0:
                selected_coordinates = coordinates[rand_select]
                noise_coords = self.noise_generator_random_configure(selected_coordinates, min_dist=self.geoid_params.min_dist, max_dist=self.geoid_params.max_dist, geoid_configure_type=self.geoid_params.geoid_configure_type)
            else:
                noise_coords = None
        
        else:
            raise NotImplementedError

        if noise_coords is not None:
            # filter unique coordinates
            noise_coords = torch.unique(noise_coords, dim=0)
            
            merged_coords = copy.deepcopy(torch.cat([coordinates, noise_coords], 0))
            q_coords, q_select, q_inverse = ME.utils.sparse_quantize(merged_coords, quantization_size=1, return_index=True, return_inverse=True)
            
            unique_coords = torch.ones_like(q_select)
            
            overlap = q_inverse[:valid_idx]
            
            unique_coords[overlap] = 0
            
            uniuqe_noise = unique_coords[q_inverse]
            uniuqe_noise = torch.where(uniuqe_noise==1)[0]
            
            
            noise_coords = merged_coords[uniuqe_noise]
            
            index = knn(coordinates[:,1:], noise_coords[:,1:], k=1, batch_x=coordinates[:,0].long(), batch_y=noise_coords[:,0].long())
            noise_feats = features[index[1]]
            
        if self.training:
            if noise_coords is not None:
                coordinates_noised = torch.cat([coordinates, noise_coords], dim=0)
                features_noised = torch.cat([features, noise_feats], dim=0)
            else:
                coordinates_noised = copy.deepcopy(coordinates)
                features_noised = copy.deepcopy(features)
            
            # generate noise aug view
            coordinates_noised[:,0] += batch['batch_size']
            
            coordinates = torch.cat((coordinates, coordinates_noised), dim=0)
            features = torch.cat((features, features_noised), dim=0)
            
            sparse_tensor = ME.SparseTensor(coordinates=coordinates.int(), features=features)
            out_seg, out_cls = self.model(sparse_tensor)
            
            pruning_mask = torch.zeros(out_seg.F.shape[0], device=self.device)
            pruning_mask[:valid_idx] = 1 # orig
            pruning_mask[:valid_idx + valid_idx] = 1 # orig of noised
            pruning_mask = pruning_mask > 0.5
            pruning = ME.MinkowskiPruning()
            out_seg = pruning(out_seg, pruning_mask.to(self.device))
            
            labels_cls = pruning_mask.float()
            
            return out_seg, out_cls, labels_cls
            
        else:
            # extend original data
            assert batch['batch_size'] == 1
            
            if noise_coords is not None:
                coordinates_noised = torch.cat([coordinates, noise_coords], dim=0)
                features_noised = torch.cat([features, noise_feats], dim=0)
            else:
                coordinates_noised = copy.deepcopy(coordinates)
                features_noised = copy.deepcopy(features)
            
            coordinates_noised[:,0] += batch['batch_size']
            
            coordinates = torch.cat((coordinates, coordinates_noised), dim=0)
            features = torch.cat((features, features_noised), dim=0)
            
            sparse_tensor = ME.SparseTensor(coordinates=coordinates.int(), features=features)
            out_seg, out_cls = self.model(sparse_tensor)

            pruning_mask = torch.zeros(out_seg.F.shape[0], device=self.device)
            pruning_mask[:valid_idx] = 1 # orig
            pruning_mask[:valid_idx + valid_idx] = 1 # orig of noised
            pruning_mask = pruning_mask > 0.5
            pruning = ME.MinkowskiPruning()
            out_seg = pruning(out_seg, pruning_mask.to(self.device))
            
            labels_cls = pruning_mask.float()
        
        return out_seg, out_cls, labels_cls
    
    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        labels = labels.repeat(2)
        
        out_seg, out_cls, labels_cls = self.forward(batch)
        preds = out_seg.F
        preds_cls = out_cls.F

        loss = 0
        seg_loss = self.criterion(preds, labels-1)
        loss += seg_loss
        
        cls_loss = self.criterion_cls(preds_cls.squeeze(1), labels_cls.float())
        loss += cls_loss
        
        self.log('seg', seg_loss.item(), prog_bar=True, logger=False, on_epoch=False)
        self.log('cls', cls_loss.item(), prog_bar=True, logger=False, on_epoch=False)
        
        return {'loss': loss, 'seg_loss': seg_loss.item(), 'cls_loss': cls_loss.item()}
    
    def training_epoch_end(self, training_step_outputs):
        
        loss = 0
        seg_loss = 0
        cls_loss = 0
        for output in training_step_outputs:
            loss += output['loss']
            seg_loss += output['seg_loss']
            cls_loss += output['cls_loss']
        
        self.log(f'train/loss', loss / len(training_step_outputs), rank_zero_only=True)
        self.log(f'train/seg_loss', seg_loss / len(training_step_outputs), rank_zero_only=True)
        self.log(f'train/cls_loss', cls_loss / len(training_step_outputs), rank_zero_only=True)
        return


