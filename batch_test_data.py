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
    parser = argparse.ArgumentParser(description="Script for batching and pickling tensors for predicting on data without labels")

    parser.add_argument('-expr', '--EXPRINPUT', default=None, type=str, required=True, help='Expression file input: tsv :)')
    parser.add_argument('-dir_out', '--DIROUT', default='./out_dir/', type=str, required=False, help='path of directory to save batches in')
    parser.add_argument('-drugs', '--DRUGS', nargs='+', default=[], required=True, help='Drugs to be pickled with the input expression')

    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    expr_file = args.EXPRINPUT
    dr = args.DIROUT
    drugs = args.DRUGS

    if not os.path.exists(dr):
        os.system('mkdir %s'%dr)

    print('reading inputs')
    read_time = time.time()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    expr = pd.read_csv(expr_file, sep='\t', index_col=0, engine='c')
    print('reading time: %s'%timer(read_time))

    for drug in drugs:
        print('running %s'%drug)
        
        drug_loader = DrugLoader(drug_list=[drug])
        
        if len(drug_loader.smiles)==0:
            print("DrugLoader failed on %s - continuing to next ..."%drug)
            continue
                        
        drug_dir = os.path.join(dr,drug)
        if not os.path.exists(drug_dir):
            os.makedirs(drug_dir)
                
        for sample in expr.index:
            expr_vec = torch.FloatTensor(expr.loc[sample].values).view(1,-1)
                
            drug_batch = Batch().from_data_list(drug_loader.get_AX())
            t = (expr_vec, drug_batch.batch, drug_batch.edge_attr, drug_batch.edge_index, drug_batch.x)
            torch.save(t, os.path.join(drug_dir,'%s.pt'%(sample)))

                
            
