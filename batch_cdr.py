#!/usr/bin/env python

import pandas as pd
import numpy as np
from torch_geometric.data import Batch
import torch
from PACE import DrugLoader, timer
import argparse
import os
from rdkit.Chem import AllChem
from rdkit import Chem
import time

if __name__ == "__main__":
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Script for batching and pickling tensors for predicting on data with labels")

    parser.add_argument('-expr', '--EXPRINPUT', default=None, type=str, required=True, help='Expression file input: tsv :)')
    parser.add_argument('-dir_out', '--DIROUT', default='./out_dir/', type=str, required=False, help='path of directory to save batches in')
    parser.add_argument('-drug_fp', '--FP', default='GCN', type=str, required=False, help='type of FP to include in the batch - GCN or MORGAN')

    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    expr_file = args.EXPRINPUT
    dr = args.DIROUT
    fp = args.FP
    
    if fp not in ['GCN', 'MORGAN']:
        raise ValueError("%s not in ['GCN', 'MORGAN']")

    if not os.path.exists(dr):
        os.system('mkdir %s'%dr)

    print('reading inputs')
    read_time = time.time()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    tcga_file = os.path.join(root_dir,'data','cdr_labels.tsv')

    cdr = pd.read_csv(tcga_file, sep='\t', index_col=0)
    expr = pd.read_csv(expr_file, sep='\t', index_col=0, engine='c')


    cdr[(cdr == 'Clinical Progressive Disease')] = 'R'
    cdr[(cdr == 'Stable Disease')] = 'R'
    cdr[(cdr == 'Complete Response')] = 'S'
    cdr[(cdr == 'Partial Response')] = 'S'
    print('reading time: %s'%timer(read_time))

        
    # preprocessing from TG-LASSO paper
    for drug in cdr.index:
        print('running %s'%drug)
        
        drug_loader = DrugLoader(drug_list=[drug])
        
        if len(drug_loader.smiles)==0:
            print('DrugLoader failed on %s ...skipping... '%drug)
            continue            
        
        r_sum = (cdr.loc[drug]=='R').sum()
        s_sum = (cdr.loc[drug]=='S').sum()

        if s_sum >=2 and r_sum>=2 and r_sum+s_sum>8:
            
            drug_dir = os.path.join(dr,drug)
            if not os.path.exists(drug_dir):
                os.makedirs(drug_dir)
                
            drug_samples = cdr.loc[drug].dropna()
            for sample in drug_samples.index:
                
                try:
                    expr_vec = torch.FloatTensor(expr.loc[sample].values).view(1,-1)
                except KeyError:
                    continue
                    
                label = cdr.loc[drug,sample]
                
                if label=='S':
                    label_enc=1
                elif label=='R':
                    label_enc=0
                label_enc = torch.LongTensor([label_enc]).view(-1,1)
                
                if fp == 'GCN':
                    drug_batch = Batch().from_data_list(drug_loader.get_AX())
                    t = (expr_vec, drug_batch.batch, drug_batch.edge_attr, drug_batch.edge_index, drug_batch.x, label_enc)
                    torch.save(t, os.path.join(drug_dir,'%s.pt'%(sample)))
                    
                elif fp == 'MORGAN':
                    sm = drug_loader.smiles[0]
                    m = Chem.MolFromSmiles(sm)
                    morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048,useChirality=True))
                    morgan_fp = torch.FloatTensor(morgan_fp).view(1,-1)
                    t = (expr_vec, morgan_fp, label_enc)
                    torch.save(t, os.path.join(drug_dir,'%s.pt'%(sample)))
        else:
            print('%s did not pass filter'%drug)
                    
                    
                    