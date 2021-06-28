from glob import glob
import numpy as np
import os
from shutil import rmtree
import h5py
from skimage.io import imread


data_path = '../DRIVE'
save_path = '../patches'
N = 1000
size = 48
seed = 42
split_ratio_train_valid=[6.5,1.5]

np.random.seed(seed)



if os.path.isdir(save_path):
    rmtree(save_path)

if not os.path.isdir(save_path):
    os.mkdir(save_path)
                

names_img_all = glob(data_path + '/training/images/*.tif')
names_mask_all = [name.replace('_training.tif','_manual1.gif').replace('images','1st_manual') for name in names_img_all]


perm=np.random.permutation(len(names_img_all))   
             
split_ind=np.array(split_ratio_train_valid)
split_ind=np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(names_img_all))).astype(np.int)


ind = {}

ind['train'] = perm[:split_ind[0]]
ind['valid']  = perm[split_ind[0]:]



for data_type in ['train','valid']:

    
    outfile = save_path + '/drive_patches_' + data_type + '.hdf5'
    
    names_img = [names_img_all[k] for k in ind[data_type]]
    names_mask = [names_mask_all[k] for k in ind[data_type]]
    
    with h5py.File(outfile,"w") as f:
        
        dset_img = f.create_dataset('imgs', (size,size,N*len(names_img)), dtype='u1')
        dset_mask = f.create_dataset('masks', (size,size,N*len(names_img)), dtype='?')
        
        slice_ = -1
    
        for img_num,(name_img,name_mask) in enumerate(zip(names_img,names_mask)):
            
        
            img = imread(name_img)
            mask = imread(name_mask)
            
            img = img[:,:,1]
            maks = mask > 0
            
        
            
            
            in_size = img.shape
            out_size = [size,size]
            
            for patch_num in range(N):
                print(str(img_num) + '    ' + str(patch_num))
                
                r1=np.random.randint(in_size[0]-out_size[0])
                r2=np.random.randint(in_size[1]-out_size[1])
                r=[r1,r2]
                
                img_patch = img[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
                maks_patch = maks[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
                
                slice_ = slice_ + 1
                dset_img[:,:,slice_] = img_patch
                dset_mask[:,:,slice_] = maks_patch
                
            
            
        

    
    
  