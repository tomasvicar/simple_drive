from glob import glob
import os
from skimage.io import imread
from skimage.io import imsave
import numpy as np
import torch 
from scipy.signal import convolve2d 
from sklearn.metrics import roc_auc_score
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image

def test_(model_name,test_path):
    
    model=torch.load(model_name)
    model.eval()
    
    
    
    names_img = glob(test_path + '/images/*.tif')

    names_mask1 = [name.replace('_test.tif','_manual1.gif').replace('images','1st_manual') for name in names_img] 
    names_mask2 = [name.replace('_test.tif','_manual2.gif').replace('images','2nd_manual') for name in names_img]
    names_mask = names_mask1 + names_mask2
    
    names_img = names_img.copy() + names_img.copy()
    
    
    names_fov =  [name.replace('_test.tif','_test_mask.gif').replace('images','mask') for name in names_img] 
    
    
    
    
    save_folder = model.config.results_folder + '/' + model.config.method + 'examples'
    
    device = torch.device(model.config.device)
    
    model=model.to(device)
    
    save_folder
    
    patch_size = model.config.patch_size  ### larger->faster, but need more ram (gpu ram)
    border = 11
    
    
    weigth_window=2*np.ones((patch_size,patch_size))
    weigth_window=convolve2d(weigth_window,np.ones((border,border))/np.sum(np.ones((border,border))),'same')
    weigth_window=weigth_window-1
    weigth_window[weigth_window<0.01]=0.01
    
    
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    accs = [] 
    aucs = []
    dices = []
    tps = []
    fps = []
    fns = []
    tns = []
    
    
    
    for img_num,(name_img, name_mask, name_fov) in enumerate(zip(names_img, names_mask, names_fov)):
        
        
        name_save = save_folder + '/' + os.path.split(name_img)[1]
        
        
        # mask = imread(name_mask)>0
        mask = np.array(Image.open(name_mask))>0
        
        fov = imread(name_fov) > 0
     
        img = imread(name_img)
        
        
        img_size=img.shape
    
        
        sum_img=np.zeros(img_size[0:2])
        count_img=np.zeros(img_size[0:2])
    
        
    
        corners=[]
        cx=0
        while cx<img_size[0]-patch_size: 
            cy=0
            while cy<img_size[1]-patch_size:
                
                corners.append([cx,cy])
                
                cy=cy+patch_size-border
            cx=cx+patch_size-border
           
        cx=0
        while cx<img_size[0]-patch_size:
            corners.append([cx,img_size[1]-patch_size])
            cx=cx+patch_size-border
            
        cy=0
        while cy<img_size[1]-patch_size:
            corners.append([img_size[0]-patch_size,cy])
            cy=cy+patch_size-border   
            
        corners.append([img_size[0]-patch_size,img_size[1]-patch_size])
        
        for corner in corners:
            
            subimg = img[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size,:]
            subimg = subimg[:,:,1]
    
            clahe = cv2.createCLAHE(clipLimit=model.config.clahe_clip,tileGridSize=(model.config.clahe_grid,model.config.clahe_grid))
            subimg = clahe.apply(subimg)

            subimg = subimg.astype(np.float64)/255 - 0.5
            subimg =  np.expand_dims(subimg,2)

            subimg=torch.from_numpy(np.transpose(subimg,(2,0,1)).astype(np.float32))
            
            subimg = subimg.unsqueeze(0)

            subimg=subimg.to(device)
        
            res=model(subimg)
            
            res=torch.sigmoid(res).detach().cpu().numpy()
            
            
            
            sum_img[corner[0]:(corner[0]+patch_size),corner[1]:(corner[1]+patch_size)]=sum_img[corner[0]:(corner[0]+patch_size),corner[1]:(corner[1]+patch_size)]+res*weigth_window
    
            count_img[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size]=count_img[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size]+weigth_window
        
        final=sum_img/count_img
        
        X = (final>0.5).astype(np.float64)[fov==1]
        X_nonbinar = final.astype(np.float64)[fov==1]
        Y = (mask>0).astype(np.float64)[fov==1]
        
        TP = np.sum(((X==1)&(Y==1)).astype(np.float64))
        FP = np.sum(((X==1)&(Y==0)).astype(np.float64))
        FN = np.sum(((X==0)&(Y==1)).astype(np.float64))
        TN = np.sum(((X==0)&(Y==0)).astype(np.float64))
        
        dice = (2 * TP )/ ((2 * TP) + FP + FN)
        
        acc = (TP+TN) / (TP + FP + FN + TN)
        
        auc = roc_auc_score(Y,X_nonbinar)
        
        
        dices.append(dice)
        
        
        accs.append(acc)
        aucs.append(auc)
        dices.append(dice)
        tps.append(TP)
        fps.append(FP)
        fns.append(FN)
        tns.append(TN)
        
        
        imsave(name_save,(final*255).astype(np.uint8))
        
        
        
        
        
    return accs,aucs,dices,tps,fps,fns,tns