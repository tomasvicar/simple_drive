from train import train
from test_ import test_
import os
from config import Config
import numpy as np
import json

if __name__ == "__main__":
    
    
    config = Config()
    
    
    if not os.path.isdir(config.best_models_dir):
        os.mkdir(config.best_models_dir)
        
    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)
        
    if not os.path.isdir(config.results_folder):
        os.mkdir(config.results_folder)
    
    
    
    # model_name = train(config)
    model_name = 'simple_drive_25_0.00100_gpu_1.36633_train_0.08977_valid_0.08799.pt'
    
    accs,aucs,dices,tps,fps,fns,tns = test_(model_name,test_path=config.test_path)

    print(np.mean(accs))
    print(np.mean(aucs))
    
    results = {}
    
    results['mean ACC'] = np.mean(accs)
    results['mean AUC'] = np.mean(accs)
    results['ACC'] = accs
    results['AUC'] = aucs
    results['DICE'] = dices
    results['TP'] = tps
    results['FP'] = fps
    results['FN'] = fns
    results['TN'] = tns
    
    

    
    

    with open('../result.json', 'w') as outfile:
        json.dump(results, outfile)    




