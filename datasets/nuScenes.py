import os
import os.path as osp

import numpy as np
import yaml
import pickle
from nuscenes import NuScenes

from .custom import CustomDataset


class nuScenesDataset(CustomDataset):

    def __init__(self,
                 data_path,
                 ignore_label=-100,
                 label_mapping=None,
                 max_volume_space=[50., 50., 3.],
                 min_volume_space=[-50., -50., -5.],
                 voxel_size=0.05,
                 beam=32,
                 fov=[-30.67, 10.67],
                 training=False,
                 use_ref=True,
                 use_instance=False,
                 test_time_adaptation=False,
                 batch_size_tta=4,
                 **kwargs,
                 ):
        super(nuScenesDataset, self).__init__(data_path, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                              voxel_size, beam, fov, training, use_ref, use_instance,
                                              test_time_adaptation=test_time_adaptation,
                                              batch_size_tta=batch_size_tta,
                                              **kwargs,
                                              )
        with open(osp.join(data_path, f'nuscenes_infos_{self.imageset}.pkl'), 'rb') as pk:
            data = pickle.load(pk)

        with open(osp.join(label_mapping), 'r') as f:
            nuscenesyaml = yaml.safe_load(f)
        self.learning_map = nuscenesyaml['learning_map']

        self.im_idx = data['infos']

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        info = self.im_idx[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        points = np.fromfile(os.path.join(self.nusc.dataroot, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        if self.use_instance:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('panoptic', lidar_sd_token)['filename'])

            points_label = np.load(lidarseg_labels_filename)['data'].reshape([-1, 1])
            sem_label = (points_label // 1000).astype(np.uint8)
            inst_label = (points_label % 1000).astype(np.uint8)
            sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

            data = (points[:,:3], points[:,3][:,None]/255, sem_label.astype(np.uint8), inst_label.astype(np.uint8))
        else:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

            data = (points[:,:3], points[:,3][:,None]/255, points_label.astype(np.uint8))

        return self.getitem(index, lidar_path, data, None)


class nuScenes_C_Dataset(CustomDataset):

    def __init__(self,
                 data_path,
                 data_path_orig,
                 ignore_label=-100,
                 label_mapping=None,
                 max_volume_space=[50., 50., 3.],
                 min_volume_space=[-50., -50., -5.],
                 voxel_size=0.05,
                 beam=32,
                 fov=[-30.67, 10.67],
                 training=False,
                 use_ref=True,
                 use_instance=False,
                 test_time_adaptation=False,
                 batch_size_tta=4,
                 corr_type='',
                 corr_level='',
                 nusc_cache=None,
                 **kwargs,
                 ):
        super(nuScenes_C_Dataset, self).__init__(data_path_orig, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                              voxel_size, beam, fov, training, use_ref, use_instance,
                                              test_time_adaptation=test_time_adaptation,
                                              batch_size_tta=batch_size_tta,
                                              **kwargs,
                                              )
        with open(osp.join(data_path_orig, f'nuscenes_infos_{self.imageset}.pkl'), 'rb') as pk:
            data = pickle.load(pk)

        with open(osp.join(label_mapping), 'r') as f:
            nuscenesyaml = yaml.safe_load(f)
        self.learning_map = nuscenesyaml['learning_map']

        self.im_idx = data['infos']

        if nusc_cache is None:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
            with open('nusc_index.pkl', 'wb') as f:
                pickle.dump(self.nusc, f)
        else:
            self.nusc = nusc_cache

        self.corr_data_dir = data_path
        self.corr_type = corr_type
        self.corr_level = corr_level

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        info = self.im_idx[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        points = np.fromfile(os.path.join(self.corr_data_dir, self.corr_type, self.corr_level, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        if self.use_instance:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('panoptic', lidar_sd_token)['filename'])

            points_label = np.load(lidarseg_labels_filename)['data'].reshape([-1, 1])
            sem_label = (points_label // 1000).astype(np.uint8)
            inst_label = (points_label % 1000).astype(np.uint8)
            sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

            data = (points[:,:3], points[:,3][:,None]/255, sem_label.astype(np.uint8), inst_label.astype(np.uint8))
        else:
            lidarseg_labels_filename = os.path.join(self.corr_data_dir, self.corr_type, self.corr_level, self.nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

            data = (points[:,:3], points[:,3][:,None]/255, points_label.astype(np.uint8))

        return self.getitem(index, os.path.join(self.corr_type, self.corr_level, lidar_path), data, None)
