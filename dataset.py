from torch.utils import data
import numpy as np
import torch 
import h5py
import matplotlib.pyplot as plt
import cv2
import time

class Dataset(data.Dataset):


    def __init__(self, augment,config,data_type):
        
        self.augment = augment
        self.config = config
        self.data_type = data_type
        self.h5data = None
        
        if data_type == 'train': 
            self.h5data_file = config.data_train
            
        elif data_type =='valid':
            self.h5data_file =  config.data_valid
            
        else:
            raise(Exception('wrong data type'))
        
        
    def __len__(self):
        with h5py.File(self.h5data_file,"r") as h5dat_tmp:
            return h5dat_tmp["imgs"].shape[2]



    def __getitem__(self, index):
        
        if  self.h5data is None:
            self.h5data = h5py.File(self.h5data_file, 'r')
            
        img = self.h5data["imgs"][:,:,index]
        mask = self.h5data["masks"][:,:,index]
        
        
        clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip,tileGridSize=(self.config.clahe_grid,self.config.clahe_grid))
        img = clahe.apply(img)
        
        
        r=[torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(4,(1,1)).view(-1).numpy()]
        if r[0]:
            img=np.fliplr(img)
            mask=np.fliplr(mask)
        if r[1]:
            img=np.flipud(img)
            mask=np.flipud(mask) 
        img=np.rot90(img,k=r[2]) 
        mask=np.rot90(mask,k=r[2])   
            
            
        img = img.astype(np.float64)/255 - 0.5
        img =  np.expand_dims(img,2)
        

        mask =  np.expand_dims(mask,2)
        mask=torch.from_numpy(np.transpose(mask,(2,0,1)).astype(np.float32))
        
        img=torch.from_numpy(np.transpose(img,(2,0,1)).astype(np.float32))
        

        
        return img,mask

        
  

if __name__ == "__main__":
    
    from config import Config    
    
    config = Config()
    config.train_num_workers = 4

    
    train_generator = Dataset(augment=True,config=config,data_type='train')
    train_generator = data.DataLoader(train_generator,batch_size=config.train_batch_size,num_workers=config.train_num_workers, shuffle=True,drop_last=True)
    
    
    start = time.time()
    for it,(img,mask) in enumerate(train_generator):
        
        # plt.imshow(np.transpose(img[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        # plt.show()
        # plt.imshow(np.transpose(mask[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        # plt.show()

        if it%10 == 0:
            end = time.time()
            print(end - start)
            start = time.time()
        
    

        
        
        
        
    
    