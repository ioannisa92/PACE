# Author: Ioannis Anastopoulos

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from utils import EarlyStopping, aws_upload, timer
from loss_util import MMD_loss
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import itertools
import time
import os



class PACE(nn.Module):
    def __init__(self, expr_dim = None, 
                 drug_dim = None,
                 lr = 1e-4, 
                 device = torch.device('cpu'), 
                 merger_layers = 1, 
                 dropout = 0.35, 
                 drug_nodes = 200, 
                 fp = 'GCN'):
        
        super(PACE, self).__init__()
        self.lr = lr
        self.device = device
        self.fp = fp
        
        # encoder (EM)
        self.EM = nn.Sequential(
                                            nn.BatchNorm1d(expr_dim),
                                            nn.Linear(expr_dim, 1024),
                                            nn.BatchNorm1d(1024), 
                                            nn.ReLU(),
                                            nn.Dropout(dropout),
                                             

                                            nn.Linear(1024, 100), 
                                            nn.BatchNorm1d(100),
                                            nn.ReLU(),
                                            nn.Dropout(dropout)
                                              )

    
        
        
        if fp == 'GCN':
            self.DM = nn.ModuleList()

            self.DM.append(GraphConv(drug_dim, drug_nodes))
            self.DM.append(TopKPooling(drug_nodes, ratio=0.8))
            self.DM.append(nn.BatchNorm1d(drug_nodes))
            self.DM.append(nn.Dropout(dropout)) 
            # concatenated latent space

            

            merged_dim = 100+(2*drug_nodes) # double the drug nodes bc we concatenate avg and max features
            
        elif fp=='MORGAN':
            self.DM = nn.Sequential(
                                        nn.Linear(drug_dim, drug_nodes),
                                        nn.BatchNorm1d(drug_nodes),
                                        nn.ReLU(),
                                        nn.Dropout(dropout))
            # concatenated latent space
            merged_dim = 100+drug_nodes
        
        x = merged_dim # input dim for the PM
        
        
        out_graph = []
        n_nodes=2*merged_dim
        for n in range(1,merger_layers+1):
            if n==merger_layers: # last layer
                out_graph.append(nn.Linear(x, 1)) # ic50 predictor
                break
            else:
                n_nodes = int(n_nodes/2)
                out_graph.append(nn.Linear(x, n_nodes))
                out_graph.append(nn.BatchNorm1d(n_nodes))
                out_graph.append(nn.ReLU())
                out_graph.append(nn.Dropout(dropout))
            x = n_nodes
        self.PM = nn.Sequential(*out_graph)
        


    
    def encode_expr(self, x):
        return self.EM(x)

        
    def encode_drug(self,data):
        
        if self.fp == 'GCN':
            batch, edge_attr, edge_index, x = data
            all_x = None # aggregate all x from each gcn+topkpool operation to be summed in the end

            for i, layer in enumerate(self.DM):

                if i == 0: # gcn conv layer
                    x = layer(x, edge_index)
                    x = self.DM[2](x) # batchnorm
                    x = F.relu(x)

                elif i==1: # topkpool layer
                    x, edge_index, _, batch, _, _ = layer(x, edge_index, None, batch)


                elif i==3: # dropout layer
                    x = layer(x)

                mean_max_x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

                if all_x is None:
                    all_x = mean_max_x
                else:
                    all_x = all_x + mean_max_x

            all_x = F.relu(all_x)
            
        elif self.fp == 'MORGAN':
            all_x = self.DM(data)


        return all_x
    
    
    def forward(self, *inp):
        
        if self.fp=='GCN':
            x1, x2 = inp[0], inp[1]
            drug_input = inp[2], inp[3], inp[4], inp[5]
            
        elif self.fp=='MORGAN':
            x1, x2 =inp[0], inp[1]
            
            drug_input = inp[2]

        feature1 = self.EM(x1) #ccle
        feature2 = self.EM(x2) #tcga
        
        drug_x = self.encode_drug(drug_input) #DM
        
        z1 = torch.cat([feature1, drug_x], dim=1) # ccle

        
        response_output = self.PM(z1) # response is predicted only for x1 (ccle)
        
        return feature1, feature2, response_output


    def forward_predict(self, *inp):
        
        if self.fp=='GCN':
            x1 = inp[0]
            drug_input = inp[1], inp[2], inp[3], inp[4]
            
        if self.fp=='MORGAN':
            x1 = inp[0]
            drug_input = inp[1]
        

        feature1 = self.EM(x1) #ccle

        drug_x = self.encode_drug(drug_input) #DM
        
        z1 = torch.cat([feature1, drug_x], dim=1) # response is predicted only for x1 (ccle)
        
        response_output = self.PM(z1)

        return response_output
        
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def fit(self,*loaders, lmbd = 0.05, epochs=200, patience=10):
        valid = False 
        if len(loaders) ==2:
            train_loader, valid_loader = loaders
            valid=True
        else:
            train_loader = loaders[0]
        
        MSE = nn.MSELoss()
        MMD = MMD_loss()
        opt = self.optimizer()
    
        epoch_hist = {}
        if valid:
            epoch_hist['val_loss'] = []
            epoch_hist['val_acc'] = []
        epoch_hist['response_loss'] = []
        epoch_hist['tissue_loss'] = []
        epoch_hist['tissue_accuracy'] = []
        epoch_hist['mmd_loss'] = []
        epoch_hist['epoch_loss'] = []
       
        train_ES = EarlyStopping(patience=patience, verbose=True,mode='train')
        if valid:
            valid_ES = EarlyStopping(patience=patience, verbose=True,mode='valid') 

        for epoch in range(epochs):
            response_loss = 0
            mmd_loss = 0
            epoch_loss = 0
            batch_n = 0
            epoch_time = time.time()
            for batch in train_loader:
                if self.fp == 'GCN':
                    x1 = batch[0].to(self.device)

                    x2 = batch[1].to(self.device)

                    batch_idx = batch[2].to(self.device)

                    edge_attr = batch[3].to(self.device)

                    edge_index = batch[4].to(self.device)

                    drug_x = batch[5].to(self.device) 

                    response = batch[6].to(self.device)


                    feature1, feature2, response_pred  = self.forward(x1,x2, batch_idx, edge_attr, edge_index, drug_x)


                    mmd = MMD(feature1, feature2) #adapting ccle representation (source) to tcga representation (target)


                    res_loss = MSE(response_pred, response)        
                    total_loss = res_loss + (lmbd * mmd) 

                    opt.zero_grad()   # clear gradients for next train
                    total_loss.backward()         # backpropagation, compute gradients
                    opt.step()        # apply gradients

                    response_loss += res_loss.item()
                    mmd_loss += mmd.item()
                    epoch_loss += total_loss.item()
                    batch_n +=1
                    
                elif self.fp == 'MORGAN':
                    x1 = batch[0].to(self.device)
                    x2 = batch[1].to(self.device)
                    drug_x = batch[2].to(self.device) 
                    response = batch[3].to(self.device)


                    feature1, feature2, response_pred  = self.forward(x1,x2, drug_x)


                    mmd = MMD(feature1, feature2) #regularizing ccle representation (source) to tcga representation (target)


                    res_loss = MSE(response_pred, response)        
                    total_loss = res_loss + (lmbd * mmd) 

                    opt.zero_grad()   # clear gradients for next train
                    total_loss.backward()         # backpropagation, compute gradients
                    opt.step()        # apply gradients

                    response_loss += res_loss.item()
                    mmd_loss += mmd.item()
                    epoch_loss += total_loss.item()
                    batch_n +=1
                    
                                
            
            response_loss = response_loss/batch_n
            mmd_loss = mmd_loss/batch_n
            epoch_loss = epoch_loss/batch_n # normalize for number of batches
            
            train_ES(epoch_loss)
            epoch_hist['response_loss']+= [response_loss]
            epoch_hist['mmd_loss']+= [mmd_loss]
            epoch_hist['epoch_loss']+= [epoch_loss]

            if valid:
                val_loss= self.test_model(valid_loader,validation=True)['loss']
                val_acc = self.test_model(valid_loader,validation=True)['acc']
                
                epoch_hist['val_loss']+=[val_loss]
                epoch_hist['val_acc']+=[val_acc]
                
                print('[Epoch %d] mmd_loss: %.5f| response_loss: %.5f | total_loss: %.5f | val_loss: %.5f | val_acc: %.5f | Time: %.3f %s'%(epoch+1, tissue_loss, tissue_accuracy, domain_loss, response_loss, epoch_loss, val_loss, val_acc, time_to_epoch, time_to_epoch_metric),flush=True)
                valid_ES(val_loss)
                if valid_ES.early_stop and train_ES.early_stop:
                    print("[Epoch %d] Early stopping"%(epoch+1),flush=True)
                    break
            else:
                time_to_epoch,time_to_epoch_metric = timer(epoch_time)
                
                
                print('[Epoch %d] mmd_loss: %.5f | response_loss: %.5f | total_loss: %.5f | Time: %.3f %s'%(epoch+1, mmd_loss, response_loss, epoch_loss, time_to_epoch, time_to_epoch_metric),flush=True)
                if train_ES.early_stop:
                    print("[Epoch %d] Early stopping"%(epoch+1),flush=True)
                    break
        return epoch_hist 

    def predict(self, loader, labels=False):
        all_preds = []
        if labels:
            all_labels = [] # specific for CDR prediction

        with torch.no_grad():
            for batch in loader:
                if self.fp == 'GCN':
                    x1 = batch[0].to(self.device)
                    batch_idx = batch[1].to(self.device)
                    edge_attr = batch[2].to(self.device)
                    edge_index = batch[3].to(self.device)
                    drug_x = batch[4].to(self.device)
                    if labels:
                        target = batch[5].to(self.device)

                    pred = self.forward_predict(x1, batch_idx, edge_attr, edge_index, drug_x)
                    if labels:
                        all_labels.append(target.cpu().numpy())
                    all_preds.append(pred.cpu().numpy())
                elif self.fp == 'MORGAN':
                    x1 = batch[0].to(self.device)
                    drug_x = batch[1].to(self.device)
                    if labels:
                        target = batch[2].to(self.device)

                    pred = self.forward_predict(x1, drug_x)
                    if labels:
                        all_labels.append(target.cpu().numpy())
                    all_preds.append(pred.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        if labels:
            all_labels = np.vstack(all_labels)
            return all_preds, all_labels
        else:
            return all_preds

    @classmethod
    def train_model(cls,all_dataset, epochs=200, patience=10, drug_nodes=None, dropout=None, gpu_id=None, fp='GCN', lmbd=None, lr=None, merger_layers=None):
        
        # making loader
        mask = np.arange(len(all_dataset))
        np.random.shuffle(mask) 
        all_loader = [all_dataset[idx] for idx in mask]
        
        use_cuda = torch.cuda.is_available()
        
        if use_cuda:
            if gpu_id is None:
                device = torch.device("cuda:%d"%(torch.cuda.current_device()))
            else:
                device = torch.device("cuda:%d"%(gpu_id))
        else:
            device = torch.device("cpu")
        
        expr_dim = all_dataset[0][0].shape[1]
        if fp == 'GCN':
            drug_dim = all_dataset[0][5].shape[1]
        elif fp == 'MORGAN':
            drug_dim = all_dataset[0][2].shape[1]
        

        model_list = []
        for i in range(10):
            
            print('Training model repeat%d on device %s...'%(i,device), flush=True)
            model = cls(expr_dim, drug_dim, device = device, drug_nodes=drug_nodes, dropout=dropout, fp=fp,lr=lr, merger_layers=merger_layers).to(device)
                
            model.train()
            train_dict = model.fit(all_loader, epochs=epochs, patience=patience, lmbd=lmbd)
            model_list.append(model)
            
        del all_loader # releasing loader from memory
        return model_list

    @classmethod
    def predict_response(cls, all_dataset, model_dir, drug_nodes=200,fp='GCN', gpu_id=None, merger_layers=1):
        
        # making loader
        mask = np.arange(len(all_dataset))
        all_loader = [all_dataset[i] for i in mask]
        
        use_cuda = torch.cuda.is_available()
        
        if use_cuda:
            if gpu_id is None:
                device = torch.device("cuda:%d"%(torch.cuda.current_device()))
            else:
                device = torch.device("cuda:%d"%(gpu_id))
        else:
            device = torch.device("cpu")
            
        expr_dim = all_dataset[0][0].shape[1]
        if fp=='GCN':
            drug_dim = all_dataset[0][4].shape[1]
        elif fp == 'MORGAN':
            drug_dim = all_dataset[0][1].shape[1]

        all_preds = []
            
        for repeat in range(10):
            model = cls(expr_dim, drug_dim, device = device, drug_nodes=drug_nodes, fp=fp, merger_layers=merger_layers).to(device)
            trained_model = os.path.join( model_dir, 'PACE.model%d'%(repeat))
            model.load_state_dict(torch.load(trained_model, map_location=device))
            
            model.eval()
            preds = model.predict(all_loader, labels=False)
            all_preds.append(preds)
            
        all_preds = np.hstack(all_preds).mean(axis=1).flatten()
                
        del all_loader # releasing memory
        return all_preds