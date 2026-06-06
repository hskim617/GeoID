import os
import numpy as np
import yaml

from .custom import CustomDataset


class KITTIDataset(CustomDataset):

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
                 **kwargs,
                 ):
        super(KITTIDataset, self).__init__(data_path, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                           voxel_size, beam, fov, training, use_ref, use_instance,
                                           test_time_adaptation=test_time_adaptation,
                                           batch_size_tta=batch_size_tta,
                                           **kwargs,
                                           )
        with open(label_mapping, 'r') as f:
            semkittiyaml = yaml.safe_load(f)
        if self.imageset == 'train':
            split = semkittiyaml['split']['train']
        elif self.imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif self.imageset == 'test':
            split = semkittiyaml['split']['test']
            
        self.learning_map = semkittiyaml['learning_map']
        self.learning_map_inv = semkittiyaml['learning_map_inv']
                
        self.im_idx = []
        for i_folder in split:
            self.im_idx += self.absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        return len(self.im_idx)
       
    def __getitem__(self, index):        
        scan_id = self.im_idx[index]
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            semantic_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            instance_data = annotated_data >> 16
            semantic_data = np.vectorize(self.learning_map.__getitem__)(semantic_data)

        if self.use_instance:
            data = (raw_data[:, :3], raw_data[:, 3][:,None], semantic_data.astype(np.int32), instance_data.astype(np.int32))
        else:
            data = (raw_data[:, :3], raw_data[:, 3][:,None], semantic_data.astype(np.int32))
        
        return self.getitem(index, scan_id, data, self.imageset)        


class KITTI_C_Dataset(CustomDataset):

    def __init__(self,
                 data_path,
                 data_path_orig,
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
                 corr_type='',
                 corr_level='',
                 **kwargs,
                 ):
        super(KITTI_C_Dataset, self).__init__(data_path_orig, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                           voxel_size, beam, fov, training, use_ref, use_instance,
                                           test_time_adaptation=test_time_adaptation,
                                           batch_size_tta=batch_size_tta,
                                           **kwargs,
                                           )
        with open(label_mapping, 'r') as f:
            semkittiyaml = yaml.safe_load(f)
        if self.imageset == 'train':
            split = semkittiyaml['split']['train']
        elif self.imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif self.imageset == 'test':
            split = semkittiyaml['split']['test']
            
        self.learning_map = semkittiyaml['learning_map']
        self.learning_map_inv = semkittiyaml['learning_map_inv']
                
        self.im_idx = []
        for i_folder in split:
            self.im_idx += self.absoluteFilePaths('/'.join([data_path_orig, str(i_folder).zfill(2), 'velodyne']))

        self.corr_data_dir = data_path
        self.corr_type = corr_type
        self.corr_level = corr_level

    def __len__(self):
        return len(self.im_idx)
       
    def __getitem__(self, index):        
        scan_id = self.im_idx[index]
        scan_id = os.path.join(self.corr_data_dir, self.corr_type, self.corr_level, 'velodyne', scan_id.split('/')[-1])
        
        raw_data = np.fromfile(scan_id, dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(scan_id.replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            semantic_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            instance_data = annotated_data >> 16
            semantic_data = np.vectorize(self.learning_map.__getitem__)(semantic_data)
            
        if self.use_instance:
            data = (raw_data[:, :3], raw_data[:, 3][:,None], semantic_data.astype(np.int32), instance_data.astype(np.int32))
        else:
            data = (raw_data[:, :3], raw_data[:, 3][:,None], semantic_data.astype(np.int32))
        
        return self.getitem(index, scan_id, data, self.imageset)
