import numpy as np
import os

class Config:
    
    data_train = "../patches/drive_patches_train.hdf5"
    data_valid = "../patches/drive_patches_valid.hdf5"
    
    model_save_dir='../tmp'
    
    best_models_dir='../best_models'
    
    results_folder = '../results'
    
    device ='cuda:0'
    
    method = 'simple_drive'

    
    init_lr = 1e-3
    lr_changes_list = np.cumsum([50,25,10,5])
    
    gamma = 0.1
    max_epochs = lr_changes_list[-1]
    
    patch_size  = 48
    
    
    filters = 32
    drop_out = 0
    depth = 3
    
    
    train_batch_size = 16
    train_num_workers = 8
    valid_batch_size = 4
    valid_num_workers = 2
    
    
    clahe_grid = 8 
    clahe_clip = 2
    
    in_channels = 1
    
    weight_decay = 1e-5

    