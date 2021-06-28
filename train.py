import os
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from shutil import copyfile
import segmentation_models_pytorch as smp
from shutil import rmtree

from dataset import Dataset
from utils.log import Log
from utils.gpu_logger import GpuLogger
from utils.training_fcns import l1_loss,l2_loss,dice_loss_logit,bce_logit
from utils.get_dice import get_dice
from unet import Unet




def train(config):
    
    gpuLogger = GpuLogger()
    device = torch.device(config.device)
    
    
    
    train_generator = Dataset(augment=True,config=config,data_type='train')
    train_generator = data.DataLoader(train_generator,batch_size=config.train_batch_size,num_workers= config.train_num_workers, shuffle=True,drop_last=True)

    valid_generator = Dataset(augment=False,config=config,data_type='valid')
    valid_generator = data.DataLoader(valid_generator,batch_size=config.valid_batch_size, num_workers=config.valid_num_workers, shuffle=True,drop_last=False)
    
    
    
    
    model = Unet(filters=config.filters,in_size=config.in_channels,out_size=1,do=config.drop_out,depth=config.depth)
    model.config = config
    model = model.to(device)
    
    model.log = Log(names=['loss','dice'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    