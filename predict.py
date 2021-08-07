import os
import argparse
import numpy as np
from PACE.utils import str2bool
from PACE.pace_model import PACE
from PACE import InMemoryBatchDataset
import pandas as pd
import torch
 

        
if __name__ == "__main__":
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Apply 6 targets drugs on all of TCGA")
    parser.add_argument('-i', '--INPUT', default=None, type=str, required=True, help='input dir to predict on')
    parser.add_argument('-o', '--OUT', default='out.npy', type=str, required=False, help='filename of results')
    parser.add_argument('-model_dir', '--MODELDIR', default='./saved_models/', type=str, required=False, help='directory of saved models')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    
    test_dir = args.INPUT
    res_out = args.OUT
    model_dir = args.MODELDIR
        
    
    dfs = []
    
    for drug in os.listdir(test_dir):
        print('Running %s'%drug, flush=True)
        
        samples = [x.split('.')[0] for x in os.listdir(os.path.join(test_dir, drug))] #samples should be same for each drug
        
        dataset = InMemoryBatchDataset(os.path.join(test_dir, drug))
        
        preds = PACE.predict_response(dataset, model_dir)
        df = pd.DataFrame(preds, index=samples, columns=[drug])
        dfs += [df.sort_index()]
        
    
    dfs = pd.concat(dfs, axis=1)
    
    print('saving results...', flush=True)
    dfs.to_csv(os.path.join( 'results', 'PACE_'+res_out), sep='\t')
    
    
                

            



