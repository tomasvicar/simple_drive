from torch.utils import data
import numpy as np
import torch 
import h5py
import atexit


class Dataset(data.Dataset):


    def __init__(self, augment,config,data_type):
        
        self.augment = augment
        self.config = config
        self.data_type = data_type
        
        if data_type == 'train': 
            self.h5data_file = config.data_train
            
        elif data_type =='valid':
            self.h5data_file =  config.data_valid
            
        else:
            raise(Exception('wrong data type'))
        
        
    def __len__(self):
        
            return 150 # self.h5dat["imgs"].shape[2]



    def __getitem__(self, index):
        
        
        img = self.h5data[:,:,index]
        
        img = img.astype(np.float64)/255
        
        img=torch.from_numpy(np.transpose(img,(2,0,1)).astype(np.float32))
        
        
        return img

        
    def h5py_worker_init(self):
        self.h5data = h5py.File(self.h5data_file, "r", libver="latest", swmr=True)
        atexit.register(self.cleanup)

    def cleanup(self):
        self.h5data.close()

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.h5py_worker_init()

if __name__ == "__main__":
    
    from config import Config    
    
    config = Config()
    
    config.train_num_workers = 0
    
    train_generator = Dataset(augment=True,config=config,data_type='train')
    train_generator = data.DataLoader(train_generator,batch_size=config.train_batch_size,num_workers=config.train_num_workers, shuffle=True,drop_last=True,worker_init_fn=Dataset.worker_init_fn)
    
    
    for img in train_generator:
        
        print(img)
    
    