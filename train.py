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
    
    model = Unet(filters=config.filters,in_size=config.in_channels,out_size=1,do=config.drop_out,depth=config.depth)
    
    
    
    
    
    
    
    
    