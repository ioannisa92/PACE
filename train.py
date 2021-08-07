import os
import argparse
import numpy as np
from PACE import str2bool
from PACE import PACE
from PACE import InMemoryBatchDataset
import torch
import itertools

        

        
if __name__ == "__main__":
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Train PACE")
    parser.add_argument('-train_dir', '--TRAINDIR', default=None, type=str, required=True, help='train directory containing all data')
    parser.add_argument('-model_dir', '--MODELDIR', default=None, type=str, required=True, help='Directory to save PACE models')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    
    train_dir = args.TRAINDIR
    model_dir = args.MODELDIR
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    root_dir = os.path.dirname(os.path.abspath(__file__))
        
    print('Training PACE...', flush=True)
            
    train_dataset = InMemoryBatchDataset(train_dir)
    model_list = PACE.train_model(train_dataset, 
                                     epochs=1, #200 
                                     patience=10, 
                                     drug_nodes=200, 
                                     dropout=0.3, 
                                     lr=1e-4,
                                     fp='GCN',
                                     lmbd=0.01, 
                                     merger_layers=1)

    print('Saving models')
    for i,model in enumerate(model_list):
        model = model.to(torch.device('cpu')).state_dict()
        model_fn='PACE.model%d'%(i)
        torch.save(model, os.path.join(root_dir, model_dir, model_fn ))
        


