 #!/usr/bin/env python

from PACE import DrugLoader, PACEDataset, da_collate, timer

import pandas as pd
import numpy as np
from torch.utils import data
import time
import torch

import os
import argparse


if __name__ == "__main__":
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Script for batching and pickling tensors. Script will output .pt files to dir_out")

    parser.add_argument('-batch_size', '--BATCHSIZE', default=500, type=int, required=False, help='size to batch input file')
    parser.add_argument('-cl_expr', '--CLEXPRINPUT', default=None, type=str, required=True, help='Cell line expression file input: tsv :)')
    parser.add_argument('-tcga_expr', '--TCGAEXPRINPUT', default=None, type=str, required=True, help='TCGA expression file input: tsv :)')
    parser.add_argument('-cl_labels', '--CLLABELS', default=None, type=str, required=True, help='Tissue labels for cell lines')
    parser.add_argument('-tcga_labels', '--TCGALABELS', default=None, type=str, required=True, help='Tissue labels for cells and tcga')
    parser.add_argument('-cdi', '--CDI', default=None, type=str, required=True, help='Cell line Drug IC50 (CDI) pairs: tsv :)')
    parser.add_argument('-dir_out', '--DIROUT', default='./out_dir/', type=str, required=False, help='path of directory to save batches in')
    parser.add_argument('-drug_fp', '--FP', default='GCN', type=str, required=False, help='type of FP to include in the batch')
    parser.add_argument('-num_workers', '--WORKERS', default=1, type=int, required=False, help='Number of workers for DataLoader to produce batch files')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########

    batch_size = args.BATCHSIZE
    ccle_expr = args.CLEXPRINPUT
    tcga_expr = args.TCGAEXPRINPUT
    ccle_labels = args.CLLABELS
    tcga_labels = args.TCGALABELS
    cdi = args.CDI
    dr = args.DIROUT
    fp = args.FP
    
    print('reading inputs', flush=True)
    read_time = time.time()
    ccle_expr = pd.read_csv(ccle_expr, sep='\t', index_col=0, engine='c')
    tcga_expr = pd.read_csv(tcga_expr, sep='\t', index_col=0, egnien='c')
    
    ccle_labels = pd.read_csv(ccle_labels, sep='\t', index_col=0)
    tcga_labels = pd.read_csv(tcga_labels, sep='\t', index_col=0)
    
    cdi = pd.read_csv(cdi, sep='\t', index_col=0)
    print('reading time: %s'%timer(read_time))
    
    if not os.path.exists(dr):
        os.system('mkdir %s'%dr)
    
    root_dir = os.path.dirname(os.path.abspath(__file__))

    print('Starting batcher...')
    if fp=='GCN':

        dataset = PACEDataset(ccle_expr, tcga_expr,ccle_labels, tcga_labels, cdi)

        loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn = da_collate)


        starttime = time.time()
        batch_count = 0
        for dat in loader:
            t = (dat[0],dat[1],dat[2],dat[3],dat[4],dat[5],dat[6])
            torch.save(t, dr+'batch%d.pt'%(batch_count))
            batch_count+=1
        print('batching time %s'%timer(starttime))
        
    elif fp == 'MORGAN':
        morgan = np.load('./data/GDSC_ic50_MorganFP.npy', allow_pickle=True).item()
        dataset = PACEDataset(ccle_expr, tcga_expr,ccle_labels, tcga_labels, cdi, morganfp=morgan)
        loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn = da_collate)


        starttime = time.time()
        batch_count = 0
        for dat in loader:
            t = (dat[0], dat[1], dat[2], dat[3])
            torch.save(t, dr+'batch%d.pt'%(batch_count))
            batch_count+=1
        print('batching time %s'%timer(starttime))
    
    