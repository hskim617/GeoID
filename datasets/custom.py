import os
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME

import torch
import math
import copy

class CustomDataset(Dataset):
    def __init__(self,
                 data_path,
                 ignore_label=-100,
                 label_mapping=None,
                 max_volume_space=[50., 50., 2.],
                 min_volume_space=[-50., -50., -4.],
                 voxel_size=0.05,
                 beam=64,
                 fov=[-23.6, 3.2],
                 training=False,
                 use_ref=True,
                 use_instance=False,
                 test_time_adaptation=False,
                 batch_size_tta=4,
                 **kwargs
                 ):
        
        self.data_root = data_path
        self.training = training
        self.imageset = 'train' if training else 'val'
        self.ignore_label = ignore_label
        self.label_mapping = label_mapping
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.voxel_size = voxel_size
        self.beam = beam
        self.fov_info = fov
        
        self.use_ref = use_ref
        self.use_instance = use_instance
        
        # tta parameters
        self.test_time_adaptation = test_time_adaptation
        self.batch_size_tta = batch_size_tta
        
        self.prefix = None
        self.suffix = None

        if self.test_time_adaptation:
            print("Test Time Training")
        else:
            print("Source Training")

    def getitem(self, index, scan_id, data, seq_name):
        if self.use_instance:
            assert len(data) == 4, "data tuple should be (xyz, ref, sem_label, inst_label)"
        else:
            assert len(data) == 3, "data tuple should be (xyz, ref, label)"

        if self.training:
            data = self.transform_train(*data)
        elif self.test_time_adaptation:
            data_list = []
            
            # if self.batch_size_tta > 1, use transforms to make augmented data
            for i in range(self.batch_size_tta - 1):
                data_list.append(copy.deepcopy(data))
            
            # different transformation for each positive sample
            data_list = [self.transform_train(*data_) for data_ in data_list]
        else:
            data = self.transform_test(*data)
        
        if data is None:
            return None

        if self.use_instance:
            xyz, ref, semantic_label, instance_label = data
        else:
            xyz, ref, semantic_label = data
        
        
        if self.test_time_adaptation:
            # prepend original test data at first
            xyz_list = [xyz]
            label_list = [semantic_label]
            
            # extend the augmented views
            for data_ in data_list:
                if self.use_instance:
                    xyz_, _, label_, _ = data_
                else:
                    xyz_, _, label_ = data_
                
                if not isinstance(xyz_, list):
                    xyz_= [xyz_]
                    label_ = [label_]
                
                xyz_list.extend(xyz_)
                label_list.extend(label_)

            # sparse quantize the coordinates, features, and labels
            coords, feats, semantic_labels, selected_indices, inverse_maps = [], [], [], [], []
            
            for i in range(len(xyz_list)):
                coord = torch.from_numpy(xyz_list[i])
                if self.use_ref:
                    feat = torch.from_numpy(ref)
                else:
                    feat = torch.ones((coord.shape[0], 1)).float()
                label = torch.from_numpy(label_list[i]).int()
                
                quantized_coord, quantized_feat, quantized_label, selected_index, inverse_map = ME.utils.sparse_quantize(coord,
                                                                                                        feat,
                                                                                                        labels=label,
                                                                                                        ignore_label=self.ignore_label,
                                                                                                        quantization_size=self.voxel_size,
                                                                                                        return_index=True,
                                                                                                        return_inverse=True)
                coords.append(quantized_coord)
                feats.append(quantized_feat)
                if i == 0: # for original data, use raw label data.
                    semantic_labels.append(label[:,0])
                else:
                    semantic_labels.append(quantized_label)
                
                selected_indices.append(selected_index)
                inverse_maps.append(inverse_map)
            
            return (scan_id, coords, feats, semantic_labels, selected_indices, inverse_maps, torch.tensor(index), seq_name, torch.from_numpy(xyz_list[0])) # add xyz coordinates
        
        if not self.training:
            coord = torch.from_numpy(xyz)
            if self.use_ref:
                feat = torch.from_numpy(ref)
            else:
                feat = torch.ones((coord.shape[0], 1)).float()
            label = torch.from_numpy(semantic_label).int()
                        
            quantized_coord, quantized_feat, quantized_label, selected_index, inverse_map = ME.utils.sparse_quantize(coord,
                                                                                                                     feat,
                                                                                                                     labels=label,
                                                                                                                     ignore_label=self.ignore_label,
                                                                                                                     quantization_size=self.voxel_size,
                                                                                                                     return_index=True,
                                                                                                                     return_inverse=True)
            return (scan_id, quantized_coord, quantized_feat, label[:,0], inverse_map, torch.tensor(index))
                

        if not isinstance(xyz, list):
            xyz = [xyz]
            ref = [ref]
            semantic_label = [semantic_label]
        
        coords, feats, semantic_labels = [], [], []
        selected_idxs, inverse_maps = [], []
        for i in range(len(xyz)):
            coord = torch.from_numpy(xyz[i])
            
            if self.use_ref:
                feat = torch.from_numpy(ref[i])
            else:
                feat = torch.ones((coord.shape[0], 1)).float()
            label = torch.from_numpy(semantic_label[i]).int()
                        
            quantized_coord, quantized_feat, quantized_label, selected_index, inverse_map = ME.utils.sparse_quantize(coord,
                                                                                                    feat,
                                                                                                    labels=label,
                                                                                                    ignore_label=self.ignore_label,
                                                                                                    quantization_size=self.voxel_size,
                                                                                                    return_index=True,
                                                                                                    return_inverse=True)
            coords.append(quantized_coord)
            feats.append(quantized_feat)
            semantic_labels.append(quantized_label)
            selected_idxs.append(selected_index)
            inverse_maps.append(inverse_map)
        
        return (scan_id, coords, feats, semantic_labels, selected_idxs, inverse_maps, torch.tensor(index))

    def data_augment(self, xyz, flip=False, rot=False, scale_aug=False, noise_transform=False):

        # random data augmentation by rotation
        if rot:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            if isinstance(xyz, list):
                for xyz_ in xyz:
                    xyz_[:,:2] = np.dot(xyz_[:,:2],j)
            else:
                xyz[:,:2] = np.dot(xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if flip:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                if isinstance(xyz, list):
                    for xyz_ in xyz:
                        xyz_[:,0] = -xyz_[:,0]    
                else:
                    xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                if isinstance(xyz, list):
                    for xyz_ in xyz:
                        xyz_[:,1] = -xyz_[:,1]    
                else:
                    xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                if isinstance(xyz, list):
                    for xyz_ in xyz:
                        xyz_[:,:2] = -xyz_[:,:2]
                else:
                    xyz[:,:2] = -xyz[:,:2]

        if scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            if isinstance(xyz, list):
                for xyz_ in xyz:
                    xyz_[:,0] = noise_scale * xyz_[:,0]
                    xyz_[:,1] = noise_scale * xyz_[:,1]
                    # xyz_[:,2] = noise_scale * xyz_[:,2]
            else:
                xyz[:,0] = noise_scale * xyz[:,0]
                xyz[:,1] = noise_scale * xyz[:,1]
                # xyz[:,2] = noise_scale * xyz[:,2]

        if noise_transform:
            noise_translate = np.array([np.random.normal(0, 0.1, 1),
                                np.random.normal(0, 0.1, 1),
                                np.random.normal(0, 0.1, 1)]).T
            if isinstance(xyz, list):
                for xyz_ in xyz:
                    xyz_[:, 0:3] += noise_translate
            else:
                xyz[:, 0:3] += noise_translate
            
        return xyz
    
    def absoluteFilePaths(self, directory):
        for dirpath, _, filenames in os.walk(directory):
            filenames.sort()
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
      
    def transform_train(self, xyz, ref, semantic_label, instance_label=None):
        xyz = self.data_augment(xyz, True, True, True, True)
        if instance_label is None:
            return xyz, ref, semantic_label
        return xyz, ref, semantic_label, instance_label

    def transform_test(self, xyz, ref, semantic_label, instance_label=None):
        xyz = self.data_augment(xyz, False, False, False, False)
        if instance_label is None:
            return xyz, ref, semantic_label
        return xyz, ref, semantic_label, instance_label

    def collate_fn(self, batch):
        scan_ids = []
        list_d = []
        list_idx = []
        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, feat, semantic_label, _, _, idx) = data
            scan_ids.append(scan_id)
            list_d.append((coord[0], feat[0], semantic_label[0]))      
            list_idx.append(idx.view(-1,1))
            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            print(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32)(list_d)
        idx = torch.cat(list_idx, dim=0)

        return_dict = {'scan_ids': scan_ids,
                       'batch_idxs': idx,
                       'coordinates': coordinates_batch,
                       'features': features_batch,
                       'labels': labels_batch,
                       'batch_size': batch_id,
                       }
        
        return return_dict

    def collate_fn_test(self, batch):
        scan_ids = []
        list_label = []
        list_d = []
        list_inv_map = []
        list_idx = []
        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, feat, semantic_label, inverse_map, idx) = data
            scan_ids.append(scan_id)
            list_label.append(semantic_label)
            list_d.append((coord, feat, semantic_label))
            list_idx.append(idx.view(-1,1))
            list_inv_map.append(inverse_map)
        
            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            print(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32)(list_d)
        idx = torch.cat(list_idx, dim=0)
    
        return {
            'scan_ids': scan_ids,
            'batch_idxs': idx,
            'labels': list_label,
            'coordinates': coordinates_batch,
            'features': features_batch,
            'inverse_maps': list_inv_map,
            'batch_size': batch_id,
        }
     
    def collate_fn_tta_v2(self, batch):
        
        scan_ids = []
        list_idx = []
        
        list_coords = []
        list_feats = []
        list_labels = []
                        
        list_sel_idx = []
        list_inv_map = []

        batch_id = 0
        for data in batch:
            if data is None:
                continue
            scan_id, coords, feats, semantic_labels, selected_idxs, inverse_maps, idx, seq_name, xyz = data
            # scan id and idx
            scan_ids.append(scan_id)
            list_idx.append(idx.view(-1,1))
            
            # original data for test
            if batch_id == 0:
                coordinates_orig, features_orig, labels_orig = ME.utils.SparseCollation(dtype=torch.float32)([(coords[0], feats[0], semantic_labels[0])])
                inverse_maps_orig = inverse_maps[0]
                
                ## added for xyz
                xyz_orig = xyz
                
            list_d = []
            
            # sample data for tta
            for i in range(self.batch_size_tta - 1):
                list_d.append((coords[1+i], feats[1+i], semantic_labels[1+i]))
            
            if len(list_d) > 0:
                coordinates, features, labels = ME.utils.SparseCollation(dtype=torch.float32)(list_d)
                list_coords.append(coordinates)
                list_feats.append(features)
                list_labels.append(labels)
            
                list_sel_idx.append(selected_idxs[1:])
                list_inv_map.append(inverse_maps[1:])

            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            print(f'batch is truncated from size {len(batch)} to {batch_id}')

        idx = torch.cat(list_idx, dim=0)
        
        return_dict = {'scan_ids': scan_ids,
                       'scene_name': seq_name,
                       'batch_idxs': idx,
                       'batch_size': batch_id,
                       'coordinates_orig': coordinates_orig,
                       'features_orig': features_orig,
                       'labels_orig': labels_orig,
                       'inverse_maps_orig': inverse_maps_orig,
                       'xyz_orig': xyz_orig,
                       }
        
        return_dict.update({
                       'coordinates': list_coords,
                       'features': list_feats,
                       'labels': list_labels,
                       'selected_idxs': list_sel_idx,
                       'inverse_maps': list_inv_map,
                       })
        
        return return_dict
        