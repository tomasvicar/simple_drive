import numpy as np
import os

class Config:
    
    data_train = "../patches/drive_patches_train.hdf5"
    data_valid = "../patches/drive_patches_valid.hdf5"
    
    
    
    device ='cuda:0'
    

    
    init_lr = 1e-2
    lr_changes_list = np.cumsum([50,25,10,5])
    
    gamma = 0.1
    max_epochs = lr_changes_list[-1]
    
    patch_size  = 48
    
    
    filters = 32
    drop_out = 0
    depth = 3
    
    
    train_batch_size = 32
    train_num_workers = 8
    valid_batch_size = 8
    valid_num_workers = 2
    

    