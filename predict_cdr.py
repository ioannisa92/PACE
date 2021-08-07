import os
import argparse
import numpy as np
from PACE import str2bool, PACE, InMemoryBatchDataset

import torch
import time
 

    
    
def run_model(dataset, loader, repeat, model_dir,fp='GCN'):
    
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda:%d"%(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
    
    if fp=='GCN':
        drug_dim = dataset[0][4].shape[1]
    elif fp == 'MORGAN':
        drug_dim = dataset[0][1].shape[1]
        
    lmbd=0.01
    model_fn = os.path.join( model_dir, 'PACE.model%d'%(repeat))
    model = PACE(expr_dim=dataset[0][0].shape[1], 
                 drug_dim=drug_dim, 
                 device = device, 
                 fp=fp).to(device)  
    model.load_state_dict(torch.load(model_fn, map_location=device))
         

    #print(model, flush=True)

    model.eval()
    preds, labels = model.predict(loader, labels=True)
    
    return preds, labels
     

def run(data_dir,model_dir,fp):

    res_dict = {}

    res_dict['test_pred'] = []
    res_dict['test_labels'] = []
    res_dict['samples'] = os.listdir(data_dir)
    
    dataset = InMemoryBatchDataset(data_dir)

    data_mask = np.arange(len(dataset))
    loader = [dataset[i] for i in data_mask]
    
    for i in range(10):

        res = run_model(dataset, loader, i, model_dir,fp = fp)
        res_dict['test_pred'] += [res[0]]
        res_dict['test_labels'] += [res[1]]
    
    return res_dict


        
if __name__ == "__main__":
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Train GenDR on all data for each pretrain condition")
    parser.add_argument('-i', '--INPUT', default=None, type=str, required=True, help='input dir to predict on')
    parser.add_argument('-o', '--OUT', default='out.npy', type=str, required=False, help='filename of results')
    parser.add_argument('-model_dir', '--MODELDIR', default='./saved_models/', type=str, required=False, help='directory of saved models')
    parser.add_argument('-drug_fp', '--FP', default='GCN', type=str, required=False, help='type of FP to include in the batch')

    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    
    inp_dir = args.INPUT
    res_out = args.OUT
    fp = args.FP
    model_dir = args.MODELDIR    
    
    final_dict = {}

    
    for drug in os.listdir(inp_dir):
        print('Running %s'%drug, flush=True)
        final_dict[drug] = {}
        resdict = run(os.path.join(inp_dir, drug),model_dir,fp)
        final_dict[drug] = resdict
        
    print('saving results...', flush=True)
    np.save(os.path.join( 'results', res_out),final_dict)
                

            



