from train import train
# from test_ import test_
import os
from config import Config



if __name__ == "__main__":
    
    
    config = Config()
    
    
    if not os.path.isdir(config.best_models_dir):
        os.mkdir(config.best_models_dir)
        
    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)
        
    if not os.path.isdir(config.results_folder):
        os.mkdir(config.results_folder)
    
    
    
    model_name = train(config)
    
    # auc,acc = test_(model_name)






