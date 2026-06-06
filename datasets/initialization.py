from datasets.kitti import KITTIDataset, KITTI_C_Dataset
from datasets.nuScenes import nuScenesDataset, nuScenes_C_Dataset
import pickle

def get_dataset_single(dataset_name,
                       training,
                       cfg,
                       generalization_param):
    assert dataset_name in ['kitti', 'nuscenes', 'waymo', 'poss'], 'unexpected dataset_name is given.'
    
    if dataset_name == 'kitti':
        dataset = KITTIDataset(**cfg.dataset_SemKITTI, **generalization_param, **cfg, training=training)
    
    elif dataset_name == 'nuscenes':
        dataset = nuScenesDataset(**cfg.dataset_nuScenes, **generalization_param, **cfg, training=training)
    else:
        raise NotImplementedError
        
    return dataset

def get_tta_dataset_single(dataset_name,
                           training,
                           cfg,
                           generalization_param,
                           test_time_adaptation_param):
    assert dataset_name in ['kitti', 'nuscenes', 'waymo', 'poss'], 'unexpected dataset_name is given.'
    
    batch_size_tta = test_time_adaptation_param.batch_size_tta
    
    if dataset_name == 'kitti':
        dataset = KITTIDataset(**cfg.dataset_SemKITTI, **generalization_param, **cfg, test_time_adaptation=True, batch_size_tta=batch_size_tta, training=training)
    
    elif dataset_name == 'nuscenes':
        dataset = nuScenesDataset(**cfg.dataset_nuScenes, **generalization_param, **cfg, test_time_adaptation=True, batch_size_tta=batch_size_tta, training=training)
    
    else:
        raise NotImplementedError
        
    return dataset

def get_dataset_single_corr(dataset_name,
                       training,
                       cfg,
                       generalization_param,
                       corr_type, corr_level, nusc_cache):
    
    assert ('kitti_c' in dataset_name) or ('nuscenes_c' in dataset_name), 'unexpected dataset_name is given.'
    
    if dataset_name == 'kitti_c':
        dataset = KITTI_C_Dataset(**cfg.dataset_SemKITTI_C, **generalization_param, **cfg, training=training, corr_type=corr_type, corr_level=corr_level, nusc_cache=nusc_cache)
        
    elif dataset_name == 'nuscenes_c':
        dataset = nuScenes_C_Dataset(**cfg.dataset_nuScenes_C, **generalization_param, **cfg, training=training, corr_type=corr_type, corr_level=corr_level, nusc_cache=nusc_cache)

    else:
        raise NotImplementedError
        
    return dataset

def get_tta_dataset_single_corr(dataset_name,
                           training,
                           cfg,
                           generalization_param,
                           test_time_adaptation_param,
                           corr_type, corr_level, nusc_cache=None):
    assert ('kitti_c' in dataset_name) or ('nuscenes_c' in dataset_name), 'unexpected dataset_name is given.'
    
    batch_size_tta = test_time_adaptation_param.batch_size_tta

    if dataset_name == 'kitti_c':
        dataset = KITTI_C_Dataset(**cfg.dataset_SemKITTI_C, **generalization_param, **cfg, test_time_adaptation=True, batch_size_tta=batch_size_tta, training=training, corr_type=corr_type, corr_level=corr_level, nusc_cache=nusc_cache)
        
    elif dataset_name == 'nuscenes_c':
        dataset = nuScenes_C_Dataset(**cfg.dataset_nuScenes_C, **generalization_param, **cfg, test_time_adaptation=True, batch_size_tta=batch_size_tta, training=training, corr_type=corr_type, corr_level=corr_level, nusc_cache=nusc_cache)

    else:
        raise NotImplementedError
        
    return dataset

def get_dataset(cfg, debug):
    generalization_param = cfg.generalization_params.copy()
    source = generalization_param.pop('source')
    target = generalization_param.pop('target')
    
    assert source in ['kitti', 'nuscenes']
    
    training_dataset = get_dataset_single(source, training=True, cfg=cfg, generalization_param=generalization_param)
    
    validation_dataset = []
    for target_name in target:
        validation_dataset.append(get_dataset_single(target_name, training=False, cfg=cfg, generalization_param=generalization_param))
        
    if debug:
        training_dataset.im_idx = training_dataset.im_idx[:64]
        for dataset in validation_dataset:
            dataset.im_idx = dataset.im_idx[:64]
    
    return training_dataset, validation_dataset

def get_test_dataset(cfg, debug):    
    target = cfg.test_params.target
    
    generalization_param = cfg.generalization_params.copy()
    generalization_param.pop('source')
    generalization_param.pop('target')
    
    test_dataset = []
    for target_name in target:
        test_dataset.append(get_dataset_single(target_name, training=False, cfg=cfg, generalization_param=generalization_param))
        
    if debug:
        for dataset in test_dataset:
            dataset.im_idx = dataset.im_idx[:64]
    
    return test_dataset

def get_nusc_instance(nusc_cache):
    if nusc_cache is None:
        with open('nusc_index.pkl', 'rb') as f:
            nusc_cache = pickle.load(f)
    return nusc_cache

def get_test_dataset_corr(cfg, debug):    
    target = cfg.test_params.target
    assert target[0] in ['kitti_c', 'nuscenes_c']
    
    generalization_param = cfg.generalization_params.copy()
    generalization_param.pop('source')
    generalization_param.pop('target')
    
    test_param = cfg.test_params.copy()
    corr_type_list = test_param.corr_type_list
    corr_level_list= test_param.corr_level_list
    
    if target[0] in ['nuscenes_c']:
        print("Loading nuscenes-c.....", end="")
        with open('nusc_index.pkl', 'rb') as f:
            nusc_cache = pickle.load(f)
        print("finished")
    else:
        nusc_cache = None

    test_dataset = []
    for target_name in target:
        for corr_type in corr_type_list:
            for corr_level in corr_level_list:
                test_dataset.append(get_dataset_single_corr(target_name, training=False, cfg=cfg, generalization_param=generalization_param,
                                                            corr_type=corr_type, corr_level=corr_level, nusc_cache=nusc_cache))
                
    if debug:
        for dataset in test_dataset:
            dataset.im_idx = dataset.im_idx[:64]
    
    return test_dataset

def get_tta_dataset(cfg, debug):
    generalization_param = cfg.generalization_params.copy()
    generalization_param.pop('source')
    generalization_param.pop('target')

    test_time_adaptation_param = cfg.test_time_adaptation_params.copy()
    target = test_time_adaptation_param.target
    
    tta_dataset = get_tta_dataset_single(target, training=False, cfg=cfg, 
                                         generalization_param=generalization_param,
                                         test_time_adaptation_param=test_time_adaptation_param)
        
    if debug:
        tta_dataset.im_idx = tta_dataset.im_idx[:16]
    
    return tta_dataset

def get_tta_dataset_corr(cfg, debug, corr_type, corr_level):
    target = cfg.test_time_adaptation_params.target
    assert target in ['kitti_c', 'nuscenes_c']
    
    generalization_param = cfg.generalization_params.copy()
    generalization_param.pop('source')
    generalization_param.pop('target')
    
    if target in ['nuscenes_c']:
        print("Loading nuscenes-c.....", end="")
        with open('nusc_index.pkl', 'rb') as f:
            nusc_cache = pickle.load(f)
        print("finished")
    else:
        nusc_cache = None

    test_time_adaptation_param = cfg.test_time_adaptation_params.copy()
    target = test_time_adaptation_param.target
            
    tta_dataset = get_tta_dataset_single_corr(target, training=False, cfg=cfg, 
                                         generalization_param=generalization_param,
                                         test_time_adaptation_param=test_time_adaptation_param,
                                         corr_type=corr_type, corr_level=corr_level,
                                         nusc_cache=nusc_cache
                                         )
    if debug:
        tta_dataset.im_idx = tta_dataset.im_idx[::16]
    
    return tta_dataset