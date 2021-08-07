# Author: Ioannis Anastopoulos

import pubchempy as pcp
from rdkit import Chem 
from rdkit.Chem import AllChem 
import os
import sys
import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch
from sklearn.preprocessing import LabelEncoder
import collections
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import urllib
import time
from torch_geometric.data import Batch

'''
Script contains classes that are used to load and process data and save files for training
'''

def da_collate(batch):
    
    x1 = []
    x2 = []
    target = []
    drug_x = []
    GCN=False
    
    for item in batch:
        x1.append(item[0])
        x2.append(item[1])
        target.append(item[3])
        
        if isinstance(item[2], list):# graph list of torch_geo Data objects
            drug_x.append(item[2][0])
            GCN=True
        else:
            drug_x.append(item[2])
            
    x1 = torch.cat(x1, dim=0)
    x2 = torch.cat(x2, dim=0)
    target = torch.cat(target, dim=0)
    
    if GCN:
        graph_batch = Batch().from_data_list(drug_x)
        batch, edge_attr, edge_index, x = graph_batch.batch, graph_batch.edge_attr, graph_batch.edge_index, graph_batch.x
        return  x1, x2, batch, edge_attr, edge_index, x, target
    else:
        drug_x = torch.cat(drug_x, dim=0)
        return x1, x2, drug_x, target
    

def my_collate(batch):
    data = [item[0] for item in batch]
    graph_list = [item[1][0] for item in batch]
    target = [item[2] for item in batch]

    
    return torch.cat(data, dim=0), Batch().from_data_list(graph_list), torch.cat(target, dim=0)

class PACEDataset(Data.Dataset):
    def __init__(self, ccle_expr, tcga_expr,ccle_labels, tcga_labels, cdi, morganfp=None):
        self.response = cdi #(cell_lines, ic50, smiles pair)
        
        self.ccle_disease = ccle_labels.disease
        self.tcga_disease = ccle_labels.disease

        
        self.ccle_expr = ccle_expr
        self.tcga_expr = tcga_expr
        
        self.tcga_disease_indices = {disease:np.where(self.tcga_disease.values==disease)[0] for disease in self.tcga_disease.values}
        self.morganfp = morganfp
        
    def __len__(self):
        return self.response.shape[0]
    
    def __getitem__(self,i):
        
        response = self.response.iloc[i].response #ic50
        smiles = self.response.iloc[i].smiles
        if self.morganfp is None:
            drug_x = DrugLoader(smiles_list=[smiles]).get_AX() # list of torch_geo Data
        elif self.morganfp is not None:
            drug_x = self.morganfp[self.response.iloc[i]['drug']]
            drug_x = torch.FloatTensor(drug_x).view(1,-1)
            
        target = torch.from_numpy(np.array([response])).type(torch.FloatTensor).view(-1,1) #ic50
        
        sample = self.response.iloc[i].name
        x1 = self.ccle_expr.loc[sample].values # sample 1 will always be ccle
        x1_disease = self.ccle_disease.loc[sample]
        if x1_disease in self.tcga_disease_indices.keys():
            x2_idx = np.random.choice(self.tcga_disease_indices[x1_disease], replace=False)
            x2 = self.tcga_expr.iloc[x2_idx] # tcga sample of the same disease as the ccle sample


        x1 = torch.FloatTensor(x1).view(1,-1)
        x2 = torch.FloatTensor(x2).view(1,-1)


        return x1, x2, drug_x, target



class ExpressionResponseDrugDataset(Data.Dataset):
    def __init__(self, X, Y):
        self.X = X # pandas dataframe of (samples, genes)
        self.y = Y # pandas dataframe of (samples, smiles, response)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
         
        response = self.y.iloc[i].response
        sample = self.y.iloc[i].name # cell line name
        smiles = self.y.iloc[i].smiles
        
        graph_list = DrugLoader(smiles_list=[smiles]).get_AX() # list of torch_geo Data

        expr = self.X.loc[sample]

        expr = torch.from_numpy(expr.values).type(torch.FloatTensor).view(1,-1)
        y = torch.from_numpy(np.array([response])).type(torch.FloatTensor).view(-1,1)


        return (expr, graph_list, y)


class InMemoryBatchDataset(Data.Dataset):

    def __init__(self, main_dir,fold=None,fp=None):
        self.main_dir = main_dir
        if fold is not None and fp is not None:
            self.all_pairs = [x for x in os.listdir(main_dir) if '%s_fold%d'%(fp,fold) in x]
            
        elif fold is not None and fp is None:
            self.all_pairs = [x for x in os.listdir(main_dir) if 'fold%d'%(fold) in x]
        elif fold is None and fp is not None:
            self.all_pairs = [x for x in os.listdir(main_dir) if '%s'%(fp) in x]
        else:
            self.all_pairs = [x for x in os.listdir(main_dir)]
            
        self.total_pairs = len(self.all_pairs)
        self.fold = fold
        self.fp = fp
        
    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        pair_loc = os.path.join(self.main_dir, self.all_pairs[idx])
        

        return torch.load(pair_loc)





class ExpressionDrugMorganDataset(Data.Dataset):
        
    def __init__(self, X,Y, morgan):
        self.X = X # pandas dataframe of (samples, genes)
        self.y = Y # pandas dataframe of (samples, smiles, auc)
        self.morgan = morgan # dict of drug: morgan fingerprints 

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        #data = self.X.iloc[i]
        #print('data',data)

        response = self.y.iloc[i].response
        sample = self.y.iloc[i].name
        drug = self.y.iloc[i]['name']
        fp = self.morgan[drug]
        expr = self.X.loc[sample]

        expr = torch.from_numpy(expr.values).type(torch.FloatTensor)
        response = torch.from_numpy(np.array([response])).type(torch.FloatTensor)
        fp = torch.from_numpy(fp).type(torch.FloatTensor)

        #print('in dataset',data.shape, y.shape)

        return (expr, fp, response)




class DrugLoader(object):
    '''
    drug_file should be a tsv file with [Index, Name] headers
    -- Index refers to the original name of the drug from GDSC
    -- Name refers ot the corrected (cleaned up) name 
    
    Class parses the names, retrieves smiles information about each name and can return
    -- A X matrices from get_AX_matrix method
    -- all unique chars among the drugs with get_unique_chars
    -- drugs that were not found in smiles db with get_excluded_drugs
    -- graph objects for all drugs with get_graph
    -- use the list of graph objects from get_graph to plot the molecule using plot_molecule
    '''
    
    def __init__(self,smiles_list = None, drug_list=None,alphabet=None):
        '''
        drug_list: list
            list of drug names only

        drug_file: str
            path to tab delimited file containing drug names
            file can also contain drug names --> smiles
            file should have a header
        '''
        
        self.smiles_dict = dict() # drug_name : isomeric smiles

        self.excluded_drugs = set() # names without smiles associated with them
        
        if smiles_list is not None:
            for i in range(len(smiles_list)):
                self.smiles_dict[i] = smiles_list[i]
         
        if drug_list is not None:
            for drug in drug_list:
                self.smiles_dict[drug] = None
            self.get_smiles() #updates self.drug_smiles 


        self.smiles_dict = collections.OrderedDict(sorted(self.smiles_dict.items()))


        
        self.alphabet = ['As', 'B', 'Br', 'C', 'Cl', 'F', 'Hg', 'I', 'K', 'N', 'Na', 'O', 'P', 'Pt', 'S', 'Sb', 'Se', 'V', 'Zn']

        self.x_map = {'atomic_num': list(range(0, 119)),
                 'chirality': ['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER'],
                 'degree': list(range(0, 11)),
                 'formal_charge': list(range(-5, 7)),
                 'num_hs': list(range(0, 9)),
                 'num_radical_electrons': list(range(0, 5)),
                 'hybridization': [
                        'UNSPECIFIED',
                        'S',
                        'SP',
                        'SP2',
                        'SP3',
                        'SP3D',
                        'SP3D2',
                        'OTHER',],
                 'is_aromatic': [False, True],
                 'is_in_ring': [False, True],
                 }
        
        self.e_map = {'bond_type': ['misc','SINGLE','DOUBLE','TRIPLE','AROMATIC',],
                'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'],
                'is_conjugated': [False, True],
                }


    @property
    def drug_name(self):
        return list(self.smiles_dict.keys())

    @drug_name.setter
    def drug_name(self, drug_name):
        self.smiles_dict[drug_name] = None
        #self.drug_names.add(drug_name)
        self.get_smiles()

    @property
    def smiles(self):
        return list(self.smiles_dict.values())

    @smiles.setter
    def smiles(self,sm_drug_name):
        # sm_drug_name is a tuple (sm,drugname)
        #if len(sm_drug_name)!=2:
        #    raise ValueError("expecting tuple of (SMILES, drug_name)")
        
        sm, drug_name = sm_drug_name[0], sm_drug_name[1]
        #self.drug_smiles.add(sm)
        #self.alphabet = self.get_unique_chars()
        if len(drug_name) ==0:
            self.smiles_dict[np.nan] = sm
            #self.drug_names.add(np.nan)
        else:
            self.smiles_dict[drug_name] = sm
            #self.drug_names.add(drug_name)
        self.alphabet = self.get_unique_chars()

    def get_unique_chars(self):
        chars=set()
        for drug in self.smiles_dict.values():
            #for sm in self.smiles[drug]['c_smiles']:
            mol = Chem.MolFromSmiles(drug)
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                chars.add(symbol)
        chars = sorted(chars)
        return list(chars)



    def get_smiles(self):
        '''
        Function takes in a a list of drug name, and returns a dictionary of mw, canonical smiles annotations, and morgan fingerprints

        drugs for which no information is found are going to ahve empty entried from mw and c_smiles
        '''

        drug_smiles = set()

        #assert type(drugs) is list
        for drug in self.smiles_dict.keys():

            #drug_items[drug] = None
            #drug_items[drug]['mw'] = set()
            #drug_items[drug]['c_smiles'] = set()
            compounds = pcp.get_compounds(drug, 'name')
            if len(compounds) !=0:
                #self.smiles_names.add(drug) # used to be index
                compounds_smiles = []
                for comp in compounds:
                    smiles=comp.isomeric_smiles
                    #mw = comp.molecular_weight
                    if len(smiles) !=0:
                        compounds_smiles +=[smiles]
                smiles_choice =  np.random.choice(compounds_smiles,1)[0]
                self.smiles_dict[drug] = smiles_choice
                    #drug_items[drug] = smiles
                    #drug_items[drug]['mw'].add(mw)
            else:
                self.excluded_drugs.add(drug)
        new_smiles_dict = {}
        for k,v in self.smiles_dict.items():
            if v is not None:
                new_smiles_dict[k] = v
        self.smiles_dict = new_smiles_dict

        #return drug_items
    
    
    def get_graph(self, smiles, plot=True):
        '''
        Function accepts iteratble smiles
        '''
        
        graph_obj = []
        for sm in smiles:
            graph_obj+=[self.__get_compound_graph(sm, graph=True)]
        
        if plot:
            for obj in graph_obj:
                self.plot_molecule(obj)

        return graph_obj


    def get_AX(self, pad=False):


        graph_list = []


        for sm in self.smiles_dict.values():
            graph = self.featurize(sm)
            graph_list.append(graph)
            
        return graph_list
        
        

    def featurize(self,smiles):
        from torch_geometric.data import Data
        mol = Chem.MolFromSmiles(smiles)
        
        xs = []
        for atom in mol.GetAtoms():
            x = []
            alphabet=np.zeros(len(self.alphabet))
            alphabet[self.alphabet.index(atom.GetSymbol())] = 1
            x.extend(list(alphabet))
#             x.append(self.alphabet.index(atom.GetSymbol()))

            atomic_num = np.zeros(len(self.x_map['atomic_num']))
            atomic_num[self.x_map['atomic_num'].index(atom.GetAtomicNum())]
            x.extend(list(atomic_num))
#             x.append(self.x_map['atomic_num'].index(atom.GetAtomicNum()))

            chirality = np.zeros(len(self.x_map['chirality']))
            chirality[self.x_map['chirality'].index(str(atom.GetChiralTag()))] = 1
            x.extend(list(chirality))
#             x.append(self.x_map['chirality'].index(str(atom.GetChiralTag())))
            
            degree = np.zeros(len(self.x_map['degree']))
            degree[self.x_map['degree'].index(atom.GetTotalDegree())] = 1
            x.extend(list(degree))
#             x.append(self.x_map['degree'].index(atom.GetTotalDegree()))
            
            formal_charge = np.zeros(len(self.x_map['formal_charge']))
            formal_charge[self.x_map['formal_charge'].index(atom.GetFormalCharge())] = 1
            x.extend(list(formal_charge))
#             x.append(self.x_map['formal_charge'].index(atom.GetFormalCharge()))
            
            num_hs = np.zeros(len(self.x_map['num_hs']))
            num_hs[self.x_map['num_hs'].index(atom.GetTotalNumHs())] = 1
            x.extend(list(num_hs))
#             x.append(self.x_map['num_hs'].index(atom.GetTotalNumHs()))
            
            num_radical_electrons = np.zeros(len(self.x_map['num_radical_electrons']))
            num_radical_electrons[self.x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons())] = 1
            x.extend(list(num_radical_electrons))
#             x.append(self.x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
            
            hybridization = np.zeros(len(self.x_map['hybridization']))
            hybridization[self.x_map['hybridization'].index(str(atom.GetHybridization()))] = 1
            x.extend(list(hybridization))
#             x.append(self.x_map['hybridization'].index(str(atom.GetHybridization())))
            

            x.append(self.x_map['is_aromatic'].index(atom.GetIsAromatic()))
            x.append(self.x_map['is_in_ring'].index(atom.IsInRing()))
            xs.append(x)
        
        x = torch.tensor(np.vstack(xs), dtype=torch.float).view(-1, len(x))

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = []
            e.append(self.e_map['bond_type'].index(str(bond.GetBondType())))
            e.append(self.e_map['stereo'].index(str(bond.GetStereo())))
            e.append(self.e_map['is_conjugated'].index(bond.GetIsConjugated()))

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

        # Sort indices.
        if edge_index.numel() > 0:
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    smiles=smiles)
        return data

 
    def plot_molecule(self, mol_graph):
        plt.figure(figsize=(7,5))
        pos_n=nx.layout.kamada_kawai_layout(mol_graph)
        nx.draw(mol_graph
                ,pos=pos_n
                ,with_labels=True
               ,node_color='green') #should update to color each molecule differently
        plt.title('Drug Graph')
        plt.show()
        plt.close()

        
            
# class ExpressionAUCLoader:
#     # unecessary but fun to make anyway 
#     def __init__(self,dataset, batch_size=64, random_state=42, shuffle=True):
#         self.dataset = dataset
#         self.batch_size=batch_size
#         self.random_state=random_state
#         self.shuffle=shuffle
        
#         n_samples = len(self.dataset)
#         np.random.seed(self.random_state)
#         all_idx = np.arange(n_samples)
#         #print(all_idx)
#         if self.shuffle:
#             np.random.shuffle(all_idx)
#         if self.batch_size<=len(dataset):
#             self.batch_idx_list = np.array_split(all_idx,self.batch_size)
#         else:
#             self.batch_idx_list = np.array_split(all_idx, len(all_idx))
        
#     def __len__(self):
#         return len(self.batch_idx_list) # number of batches
#     def __iter__(self):
#         return self.sampler()
        
#     def sampler(self):
        
#         for batch_idx in self.batch_idx_list:
#             batch_data = []
#             batch_targets = []
#             for idx in batch_idx:
#                 batch_data.append(self.dataset[idx][0])
#                 batch_targets.append(self.dataset[idx][1])

#             batch_data = torch.cat(batch_data, dim=0)
#             batch_targets = torch.cat(batch_targets, dim=0)
#             yield batch_data, batch_targets

            
           
# class ExpressionAUCDrugLoader:
#     # necessary and fun to make
#     def __init__(self,dataset, batch_size=64, random_state=42, shuffle=True):
#         self.dataset = dataset
#         self.batch_size=batch_size
#         self.random_state=random_state
#         self.shuffle=shuffle

#         n_samples = len(self.dataset)
#         np.random.seed(self.random_state)
#         all_idx = np.arange(n_samples)
#         #print(all_idx)
#         if self.shuffle:
#             np.random.shuffle(all_idx)
        
        
#         if self.batch_size<=len(dataset):
#             self.batch_idx_list = np.array_split(all_idx,int(len(self.dataset)/self.batch_size))
#         else:
#             self.batch_idx_list = np.array_split(all_idx, len(all_idx))

#     def __len__(self):
#         return int(len(self.dataset)/self.batch_size)
#     def __iter__(self):
#         return self.sampler()

#     def sampler(self):
#         from torch_geometric.data import Batch
#         import time
#         batch_time = time.time()
#         for batch_idx in self.batch_idx_list:
#             batch_expr = []
#             batch_drugs = []
#             batch_targets = []
#             for idx in batch_idx:
#                 batch_expr.append(self.dataset[idx][0])
#                 batch_drugs.extend(self.dataset[idx][1])
#                 batch_targets.append(self.dataset[idx][2])

#             #for drug in batch_drugs:
#             #    print('from batch_drugs',drug)
#             batch_data = torch.cat(batch_expr, dim=0)
#             batch_drugs = Batch().from_data_list(batch_drugs)
#             batch_targets = torch.cat(batch_targets, dim=0)
            
#             print('Time to batch drug response data: %.3f %s'%(timer(batch_time)))
#             batch_time=time.time()
            
#             yield batch_data, batch_drugs,batch_targets

# class MorganDataset(Data.Dataset):
#     def __init__(self, smiles, y, mode='classifier'):
#         self.smiles = smiles
#         self.y = y
#         self.mode = mode

#     def __len__(self):
#         return len(self.smiles)

#     def __getitem__(self,i):
#         from rdkit.Chem import AllChem
#         from rdkit import Chem
#         sm = self.smiles[i]
#         m = Chem.MolFromSmiles(sm)
#         fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048,useChirality=True))         
#         y = self.y[i]

#         if self.mode == 'regressor':
#             y = torch.from_numpy(np.array([y])).type(torch.FloatTensor)
#         elif self.mode == 'classifier':
#             y = torch.Tensor([y]).type(torch.LongTensor)
#         fp = torch.from_numpy(fp).type(torch.FloatTensor)
#         return fp, y

# class ExpressionLoader(object):
#     '''
#     Class loads expression files from TARGET, TCGA, GDSC, Treehouse
#     Initial files for the expression for the above are in log2(TPM+1), except from GDSC (RMA normalized)

#     Class loads data in desired format
#     -- exponentially normalized
#     -- standardized
#     -- normalized 0-1
#     -- etc.

#     '''
#     def __init__(self, data):
#         cwd = os.getcwd()
        
#         self.Treehouse = cwd+'/data/Treehouse_log2TPM.tsv'
        
#         self.TARGET = cwd+'/data/TARGET_log2TPM.tsv'

#         self.GDSC = cwd+'/data/GDSC_expr_rma.tsv'

#         self.TCGA = cwd+'/data/TCGA_RNAseq_log2tpmplus1_remove_dupsamples_top10kcclegenes_forpretraining.tsv'
       
#         self.COSMIC_CCLE = cwd+'/data/COSMIC_CCLE_RNAseq_log2tpm.tsv'

#         self.CCLE = cwd+'/data/CCLE_RNAseq_rsem_genes_log2tpm_plus_1_DepMapLabels.tsv'

#         self.recount = cwd+'/data/recount_log2TPM_plus_1_top10kcclegenes_for_pretraining.tsv'

#         self.all_data = cwd+'/data/Treehouse.TARGET.CCLE.recount_top10kgenes.tsv'
        
#         if 'TARGET' in data: 
#             target_metadata_fn = cwd+'/data/TARGET_phenotype.tsv'
#             self.target_metadata = pd.read_csv(target_metadata_fn, sep='\t', index_col=0)
#         if 'Treehouse' in data:
#             th_metadata_fn = cwd+'/data/TreehousePEDv9_clinical_metadata.2019-03-15.tsv'
#             self.th_metadata = pd.read_csv(th_metadata_fn, sep='\t', index_col=0)
#         if 'TCGA' in data:
#             #tcga_metadata_fn = '/projects/sysbio/users/ianastop/data/tcga_disease.tsv' # this file is not entirely correct
#             tcga_metadata_fn = cwd+"/data/TCGA_subtype_assignments.tsv"
#             self.tcga_metadata = pd.read_csv(tcga_metadata_fn, sep='\t', index_col=0)["DISEASE"]
#         if 'GDSC' in data:
#             gdsc_metadata_fn = cwd+'/data/GDSC_Cell_Lines_Details.csv'
#             self.gdsc_metadata = pd.read_csv(gdsc_metadata_fn, sep=',', index_col=0).reset_index().set_index('COSMIC identifier')
#         if 'CCLE' in data:
#             ccle_metadata_fn = cwd+'/data/DepMapCellLineInfo.csv'
#             self.ccle_metadata = pd.read_csv(ccle_metadata_fn, sep=',', index_col=0)['disease']
#         if 'COSMIC-CCLE' in data:
#             ccle_metadata_fn = cwd+'/data/DepMapCellLineInfo.csv'
#             self.cosmic_ccle_metadata = pd.read_csv(ccle_metadata_fn, sep=',', index_col=4)['disease']
#         self.data = data
#         for d in data:
#             if d not in ["TARGET","TCGA","GDSC","Treehouse", "CCLE", 'recount','COSMIC-CCLE','all']:
#                 raise ValueError('invalid data entry. Valid entries are ["TARGET","TCGA","GDSC","Treehouse", "CCLE", "recount","COSMIC-CCLE"]. Got {}'.format(data))

#         self.metadata=[]
        
    
#     @property
#     def get_metadata(self):
#         #same order as the flags in self.data
#         return self.metadata
        
#     def load(self, norm=None, intersect_samples=False, intersect_genes=False,concat_samples=False):
#         data =[]
#         scalers = [] 
        
#         if norm not in ["minmax","standard",None]:
#             raise ValueError('invalid normalization entry. Valid entried are ["standard", "minmax"]')
#         for d in self.data:
#             if d == 'all':
#                 all_data = pd.read_csv(self.all_data, engine='c', index_col=0, sep='\t')
#                 index = all_data.index.values
#                 columns = all_data.columns.values
#                 if norm:
#                     all_data, all_data_scaler = self.normalize_data(all_data, norm)
#                     all_data = pd.DataFrame(all_data, index=index, columns=columns)
#                     data += [all_data]
#                     scalers +=[all_data_scaler]
#                 else:
#                     data +=[all_data]
            
#             if d=='recount':
#                 print('loading %s...'%d)
#                 recount = pd.read_csv(self.recount, engine='c', index_col=0, sep='\t')
#                 index = recount.index.values
#                 columns = recount.columns.values
#                 if norm:
#                     recount, recount_scaler = self.normalize_data(recount, norm)
#                     recount = pd.DataFrame(recount, index=index, columns=columns)
#                     data += [recount]
#                     scalers +=[recount_scaler]
#                 else:
#                     data +=[recount]

#             if d=='TARGET':
#                 print('loading %s...'%d)
#                 target = pd.read_csv(self.TARGET, index_col=0, sep='\t', engine='c')
                
#                 target_metadata = self.target_metadata['primary_disease_code'].loc[target.index]
#                 self.metadata+=[target_metadata]
                
#                 index = target.index.values
#                 columns = target.columns.values
                
#                 if norm:
#                     target, target_scaler = self.normalize_data(target, norm)
#                     target = pd.DataFrame(target, index=index, columns=columns) 
#                     data += [target]
#                     scalers +=[target_scaler]
    
#                 else:
#                     data +=[target]

#             if d=='TCGA':
#                 print('loading %s...'%d)
#                 tcga = pd.read_csv(self.TCGA, index_col=0, sep='\t', engine='c')
#                 #tcga_idx = list(map(lambda x:x[:-3], tcga.index.values)) #for getting metadata
               
#                 #metadata_dict = self.tcga_metadata.to_dict()['acronym']
                 
#                 #disease_list = []
#                 #for idx in tcga_idx:
#                 #    if idx in metadata_dict:
#                 #        disease = metadata_dict[idx]
#                 #        disease_list +=[disease]
#                 #    else:
#                 #        disease_list += ['NA']

#                 #tcga_metadata = pd.DataFrame(disease_list, index = tcga_idx, columns=['Disease'])
#                 common_samples = set(tcga.index).intersection(self.tcga_metadata.index)
#                 tcga = tcga.loc[common_samples]
#                 tcga_metadata = self.tcga_metadata.loc[common_samples]
#                 self.metadata+=[tcga_metadata]
               
#                 index = tcga.index.values
#                 columns = tcga.columns.values
 
#                 if norm:
#                     tcga, tcga_scaler = self.normalize_data(tcga, norm)
#                     tcga = pd.DataFrame(tcga, index=index, columns=columns)
#                     data += [tcga]
#                     scalers +=[tcga_scaler]
#                 else:
#                     data +=[tcga]

#             if d=='GDSC':
#                 print('loading %s...'%d)
#                 gdsc = pd.read_csv(self.GDSC, index_col=0, sep='\t', engine='c')
#                 sample_inter = list(set(gdsc.index).intersection(self.gdsc_metadata.index))
#                 gdsc_metadata = self.gdsc_metadata['Cancer Type\n(matching TCGA label)'].loc[sample_inter]
                
#                 self.metadata+=[gdsc_metadata.fillna('UNABLE TO CLASSIFY').drop(909744.0)] #dropping this sample, because it is one sample for one disease type (ACC)
                
#                 index = gdsc.index.values
#                 columns = gdsc.columns.values

#                 if norm:
#                     gdsc, gdsc_scaler = self.normalize_data(gdsc,norm)
#                     gdsc = pd.DataFrame(gdsc, index=index, columns=columns)
#                     data += [gdsc]
#                     scalers +=[gdsc_scaler]
#                 else:
#                     data+=[gdsc]
            
#             if d=='CCLE':
#                 print('loading %s...'%d)
#                 ccle = pd.read_csv(self.CCLE, index_col=0, sep='\t', engine='c')
#                 ccle,self.ccle_metadata = self.sample_intersection([ccle, self.ccle_metadata])
                
#                 self.metadata+=[self.ccle_metadata]

#                 index = ccle.index.values
#                 columns = ccle.columns.values

#                 if norm:
#                     ccle, ccle_scaler = self.normalize_data(ccle,norm)
#                     ccle = pd.DataFrame(ccle, index=index, columns=columns)
#                     data += [ccle]
#                     scalers +=[ccle_scaler]
#                 else:
#                     data+=[ccle]

#             if d=='COSMIC-CCLE':
#                 print('loading %s...'%d)
#                 ccle = pd.read_csv(self.COSMIC_CCLE, index_col=0, sep='\t', engine='c') #this dataframe ha already been intersected for common genes
#                 ccle,self.cosmic_ccle_metadata = self.sample_intersection([ccle, self.cosmic_ccle_metadata])

#                 self.metadata+=[self.cosmic_ccle_metadata]

#                 index = ccle.index.values
#                 columns = ccle.columns.values

#                 if norm:
#                     ccle, ccle_scaler = self.normalize_data(ccle,norm)
#                     ccle = pd.DataFrame(ccle, index=index, columns=columns)
#                     data += [ccle]
#                     scalers +=[ccle_scaler]
#                 else:
#                     data+=[ccle]

#             if d=='Treehouse':
#                 print('loading %s...'%d)
#                 treehouse = pd.read_csv(self.Treehouse, index_col=0, sep='\t', engine='c')
#                 th_metadata = self.th_metadata.disease.loc[treehouse.index]
    
#                 self.metadata+=[th_metadata]

#                 index = treehouse.index.values
#                 columns = treehouse.columns.values

#                 if norm:
#                     treehouse, treehouse_scaler = self.normalize_data(treehouse,norm)
#                     treehouse = pd.DataFrame(treehouse, index=index, columns=columns)
#                     data += [treehouse]
#                     scalers +=[treehouse_scaler]
#                 else:
#                     data+=[treehouse]

#         if intersect_samples and len(data) >=2:
#             data = self.sample_intersection(data)
#         if intersect_genes and len(data) >=2:
#             data = self.gene_intersection(data)
#         if intersect_samples and len(data) <2:
#             warnings.warn('cannot perform sample intersection with less than 2 dataframes', Warning,stacklevel=2)
#             #raise ValueError('cannot perform intersection with less than 2 dataframes')
#         if intersect_genes and len(data) <2:
#             warnings.warn('cannot perform gene intersection with less than 2 dataframes', Warning,stacklevel=2)
#             #raise ValueError('cannot perform intersection with less than 2 dataframes')
#         if concat_samples and len(data) <2:
#             warnings.warn('cannot concatenate with less than 2 dataframes', Warning,stacklevel=2)
            
#         if concat_samples and len(data)>2:
#             data = pd.concat(data,axis=0)
#         return data

#     def sample_intersection(self, dfs):
        
#         # list of dfs needs to samples by genes
#         # dfs is a list of dataframes

#         # function returns the same list of dataframes with genes intersected for all three

#         if len(dfs) <2:
#             print('cannot do intersection with less than 2 dataframes')
#             sys.exit(0)

#         all_samples=[]
#         for df in dfs:
#             all_samples+=[df.index]

#         intersection_samples=set.intersection(*map(set,all_samples))

#         #print('%d samples intersect'%len(intersection_samples))

#         intersected_dfs=[]
#         for df in dfs:

#             intersected_df=df.loc[list(intersection_samples)]
#             intersected_df = intersected_df[~intersected_df.index.duplicated(keep='first')] #removing potential duplicate indices
#             intersected_dfs+=[intersected_df]

#             #print('shape after sample intersection:')
#             #print(intersected_df.shape)

#         return intersected_dfs


#     def gene_intersection(self,dfs):
#         # list of dfs needs to samples by genes
#         # dfs is a list of dataframes

#         # function returns the same list of dataframes with genes intersected for all three

#         if len(dfs) <2:
#             print('cannot do intersection with less than 2 dataframes')
#             sys.exit(0)

#         all_genes=[]
#         for df in dfs:
#             all_genes+=[df.columns]

#         intersection_genes=set.intersection(*map(set,all_genes))

#         #print('%d genes intersect'%len(intersection_genes))

#         intersected_dfs=[]
#         for df in dfs:

#             intersected_df=df[list(intersection_genes)]
#             intersected_dfs+=[intersected_df]

#             #print('shape after gene intersection:')
#             #print(intersected_df.shape)

#         return intersected_dfs

#     def _standardize(self,data):
#         from sklearn.preprocessing import StandardScaler
#         scaler = StandardScaler()
#         t_data = scaler.fit_transform(data)
#         return t_data, scaler

#     def _minmax(self,data):
#         #normalizes (0,1)
#         from sklearn.preprocessing import MinMaxScaler
#         scaler = MinMaxScaler()
#         t_data = scaler.fit_transform(data)
#         return t_data, scaler

#     def normalize_data(self,data, norm):
#         #wrapper for normalizers
#         if norm=='standard':
#             return self._standardize(data)
#         elif norm=='minmax':
#             return self._minmax(data)


# class DAGenDRDataset(Data.Dataset):
#     def __init__(self, main_dir,fold=None,fp=None):
#         self.main_dir = main_dir
#         if fold is not None and fp is not None:
#             self.all_pairs = [x for x in os.listdir(main_dir) if '%s_fold%d'%(fp,fold) in x]
            
#         elif fold is not None and fp is None:
#             self.all_pairs = [x for x in os.listdir(main_dir) if 'fold%d'%(fold) in x]
#         elif fold is None and fp is not None:
#             self.all_pairs = [x for x in os.listdir(main_dir) if '%s'%(fp) in x]
#         else:
#             self.all_pairs = [x for x in os.listdir(main_dir)]
            
#         self.total_pairs = len(self.all_pairs)
#         self.fold = fold
#         self.fp = fp
        
#     def __len__(self):
#         return self.total_pairs

#     def __getitem__(self, idx):
#         # i think the below line is all i nedd
#         pair_loc = os.path.join(self.main_dir, self.all_pairs[idx])
        

# #         x1, x2, y1,y2, z, u, batch_idx, edge_attr, edge_index, x,  target = torch.load(pair_loc)
#         return torch.load(pair_loc)

# class InMemoryDataset(Data.Dataset):
#     # dont think is needed, GenDRDataset or DAGenDataset should work for CDR prediction
#     def __init__(self, main_dir, fp=None):
#         self.main_dir = main_dir
#         if fp is not None:
#             self.all_pairs = [x for x in os.listdir(main_dir) if '%s_'%fp in x]
#         else:
#             self.all_pairs = [x for x in os.listdir(main_dir)]
#         self.total_pairs = len(self.all_pairs)
#         self.fp = fp
        
#     def __len__(self):
#         return self.total_pairs

#     def __getitem__(self, idx):

#         pair_loc = os.path.join(self.main_dir, self.all_pairs[idx])

# #         expr, batch_idx, edge_attr, edge_index, x,  target = torch.load(pair_loc)
#         return torch.load(pair_loc)

# class SmilesTargetLoader():
#     '''
#     fn should be tab delim of smiles-->target
#     '''
#     def __init__(self,fn,mode='classifier', class_col=1, header=False, alphabet=None):
        
#         #df = pd.read_csv(fn, index_col=0, sep='\t')
#         #self.smiles = df.smiles.values
#         #self.drug_name = df.name.values
#         #sorted_name_smiles = sorted(list(zip(self.drug_name, self.smiles)))
#         #self.drug_name_unique = list(set([x[0] for x in sorted_name_smiles]))
#         #self.smiles_unique = list(set([x[1] for x in sorted_name_smiles]))
        
#         #self.y = df.auc.values
        
#         self.smiles = []
#         self.y = []
#         self.mode=mode
#         if header:
#             self.classes = []
#         if mode == 'classifier':
#             #self.classes = []
#             self.num_classes = None
#         with open(fn, 'r') as f:
#             lines = f.readlines()
#             if header is False:
#                 header_idx=0
#             else:
#                 header_idx=1
#                 self.classes.extend(lines[0].split()[class_col:])
#             for line in lines[header_idx:]:
#                 line = line.split()
#                 self.smiles.append(line[0].strip())
#                 target = line[class_col:]
#                 if mode=='classifier':
#                     target = list(map(lambda x:int(float(x.strip())), target))
#                     if len(target)==1: #binary classification
#                         self.y.append(target[0])
#                         self.num_classes=1
#                     else:
#                         self.y.append(target)
#                         self.num_classes = len(target)
#                     #self.classes.extend(target)
#                 elif mode=='regressor':
#                     target = list(map(lambda x:float(x.strip()), target))
#                     self.num_classes = len(target)
#                     self.y.append(target)

#         self.bond_types={ 'UNSPECIFIED': 0,
#                             'SINGLE':1,
#                             'DOUBLE':2,
#                             'TRIPLE':3,
#                             'QUADRUPLE':4,
#                             'QUINTUPLE':5,
#                             'HEXTUPLE':6,
#                             'ONEANDAHALF':7,
#                             'TWOANDAHALF':8,
#                             'THREEANDAHALF':9,
#                             'FOURANDAHALF':10,
#                             'FIVEANDAHALF':11,
#                             'AROMATIC':11,
#                             'IONIC':12,
#                             'HYDROGEN':13,
#                             'THREECENTER':14,
#                             'DATIVEONE':15,
#                             'DATIVE':16,
#                             'DATIVEL':17,
#                             'DATIVER':18,
#                             'OTHER':0, 'ZERO':0}

#         self.bond_stereo  = {'STEREONONE' : 0,
#                     'STEREOANY'   : 1,
#                     'STEREOZ'     : 2,
#                     'STEREOE'     : 3}

#         self.failed_smiles = [] # smiles that fail to go through rdkit
#         if alphabet is None:
#             self.alphabet = self.get_unique_chars()
#         else:
#             self.alphabet = self.get_unique_chars() # rerunning to exclude failes smiles
#             self.alphabet = alphabet
#         print('Failed SMILES: %d'%len(self.failed_smiles))
#         self.hybridization_alphabet = ['UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER']
#         self.degree_alphabet = [0,1,2,3,4,5,6]
#         self.hydrogens_alphabet = [0,1,2,3,4]
#         self.chiral_alphabet = [ 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW']
#         self.total_features = len(self.alphabet)+len(self.hybridization_alphabet)+len(self.degree_alphabet)+len(self.hydrogens_alphabet)+len(self.chiral_alphabet)+5 #4 because of formal charge, radical electrons, aromaticity, is_chiral, and mass
    
#     def get_unique_chars(self):
#         chars=set()
#         for i,sm in enumerate(self.smiles): # get_AX_torch breaks with set(self.smiles) W E I R D
#             #for sm in self.smiles[drug]['c_smiles']:
#             mol = Chem.MolFromSmiles(sm)
#             if mol is not None:
#                 for atom in mol.GetAtoms():
#                     atom_idx = atom.GetIdx()
#                     symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
#                     chars.add(symbol)
#             else:
#                 self.failed_smiles.append(sm)
#                 self.smiles.pop(i)
#                 self.y.pop(i)
            
#         chars = sorted(chars)
#         return list(chars)

#     def to_edge_index(self,m):
#         edge_index = np.vstack([np.nonzero(m)[0],np.nonzero(m)[1]])
#         return torch.from_numpy(edge_index).type(torch.LongTensor)

#     def to_edge_attr(self,edge_index,A_bond_type, A_conjugated, A_stereo):
#         rows, cols = edge_index
#         edge_attr = np.zeros((edge_index.shape[1], len(self.bond_types.keys())+1+len(self.bond_stereo.keys())))
#         for i in np.arange(len(rows.numpy())):
#             row_idx = rows[i]
#             col_idx = cols[i]
#             bond_type = A_bond_type[row_idx, col_idx]
#             stereo_type = A_stereo[row_idx, col_idx]
#             edge_attr[i,int(bond_type)] = 1
#             #edge_attr[i,len(self.bond_types.keys())+1] = A_aromatic[row_idx, col_idx] # boolean
#             edge_attr[i,len(self.bond_types.keys())+1] = A_conjugated[row_idx, col_idx] # boolean
#             edge_attr[i,len(self.bond_types.keys())+1+int(stereo_type)] = 1
            
#         return torch.from_numpy(edge_attr).type(torch.FloatTensor)

#     def get_AX_torch(self):
#         # torch_geo does not require padding
#         from torch_geometric.data import Data

#         A, X, A_bond_type, A_conjugated, A_stero  = self.get_AX_matrix()


#         data_list = []
#         for i in np.arange(len(A)):
#             a = A[i]
#             x = X[i]
#             a_bond_type = A_bond_type[i]
#             a_conjugated = A_conjugated[i]
#             a_stereo = A_stero[i]

#             edge_index = self.to_edge_index(a)
#             edge_attr  = self.to_edge_attr(edge_index, a_bond_type,a_conjugated, a_stereo)
#             x = torch.from_numpy(x).type(torch.FloatTensor)

#             y = self.y[i]# the order of the drugs shoudl be the same as in self.smiles
#             if self.mode=='classifier':
#                 y = torch.Tensor([y]).type(torch.FloatTensor)
#             elif self.mode=='regressor':
#                 y = torch.Tensor([y]).type(torch.FloatTensor)
#             data = Data(edge_index=edge_index, x=x, y=y, edge_attr=edge_attr)
#             data_list.append(data)
#         return data_list

#     def get_AX_matrix(self, pad=False):
#         '''
#         pad - makes matrices same size by adding zeros
#         keep_dups - some compounds have multiple smiles
#         '''
#         A_mat_list = []
#         X_mat_list = []

#         A_bond_type_mat_list = []
#         #A_aromatic_mat_list = []
#         A_conjugated_mat_list = []
#         A_stero_mat_list = []

#         for drug in tqdm(self.smiles, file=sys.stdout):
#             A, X, A_bond_type, A_conjugated, A_stero = self.get_compound_features(drug, graph=False)
#             A_mat_list+=[A]
#             X_mat_list+=[X]
#             A_bond_type_mat_list += [A_bond_type]
#             #A_aromatic_mat_list += [A_aromatic]
#             A_conjugated_mat_list += [A_conjugated]
#             A_stero_mat_list += [A_stero]
#         if pad:
#             padded_A_mat = self.__pad_nodes(A_mat_list)
#             padded_X_mat = self.__pad_nodes(X_mat_list, axis=0)
#             return padded_A_mat, padded_X_mat
#         else:
#             return np.array(A_mat_list), np.array(X_mat_list), np.array(A_bond_type_mat_list), np.array(A_conjugated_mat_list), np.array(A_stero_mat_list) 


#     def get_compound_features(self,smiles, graph=False):
#         '''

#         '''
        
#         mol = Chem.MolFromSmiles(smiles)
        
#         nAtms=mol.GetNumAtoms()
#         # adjency matrix (binary) indicating whcih atom is connected to each other atom
#         A            = np.zeros((nAtms, nAtms))
#         A_bond_type  = np.zeros((nAtms, nAtms)) 
#         #A_aromatic   = np.zeros((nAtms, nAtms)) # removing bc this features is already within self.bond_types
#         A_conjugated = np.zeros((nAtms, nAtms))
#         A_stero      = np.zeros((nAtms, nAtms))        

    
#         X_elements = np.zeros((nAtms, len(self.alphabet)))
#         X_hybrdization = np.zeros((nAtms, len(self.hybridization_alphabet)))
#         X_degree = np.zeros((nAtms, len(self.degree_alphabet)))
#         X_hydrogens = np.zeros((nAtms, len(self.hydrogens_alphabet)))
#         X_chiral = np.zeros((nAtms, len(self.chiral_alphabet)))
#         X_rest = np.zeros((nAtms, 5)) #4 because of formal charge, radical electrons, is_aromatic, is_chiral, and mass

#         #X_init = self.one_hot_element(smiles) # beginning the entries in the X matrix with one hot encoding elements
#         #X_init_features = X_init.shape[1]

#         #X_add = np.zeros((X_init.shape[0], 4)) # adding features [atom_degree, nH, implicit valence, aromaticity indicator]
#         #X = np.hstack([X_init, X_add])

#         for atom in mol.GetAtoms():
#             for n in (atom.GetNeighbors()):

#                 ref_atom_idx = atom.GetIdx()
#                 neighbor_atom_idx = n.GetIdx()
#                 A[ref_atom_idx,neighbor_atom_idx] = 1

#                 ####################--- BOND FEATURES--###################
#                 bond = mol.GetBondBetweenAtoms(ref_atom_idx,neighbor_atom_idx)
#                 bond_type = str(bond.GetBondType())
                
#                 A_bond_type[ref_atom_idx,neighbor_atom_idx] = self.bond_types[bond_type]
#                 #A_aromatic[ref_atom_idx, neighbor_atom_idx] = bond.GetIsAromatic()
#                 A_conjugated[ref_atom_idx, neighbor_atom_idx] = bond.GetIsConjugated()
#                 A_stero[ref_atom_idx, neighbor_atom_idx] = self.bond_stereo[str(bond.GetStereo())]
#                 ####################--- BOND FEATURES--###################
                
#                 ####################--- ATOM FEATURES--###################
#                 symbol = mol.GetAtomWithIdx(ref_atom_idx).GetSymbol()
#                 symbol_idx = self.alphabet.index(symbol)
#                 X_elements[ref_atom_idx, symbol_idx]=1

#                 formal_charge = atom.GetFormalCharge()
#                 X_rest[ref_atom_idx, -5] = formal_charge

#                 radical_electrons = atom.GetFormalCharge()
#                 X_rest[ref_atom_idx, -4]=radical_electrons

#                 hybridization_type = atom.GetHybridization()
#                 hybridization_type_idx = self.hybridization_alphabet.index(str(hybridization_type))
#                 X_hybrdization[ref_atom_idx, hybridization_type_idx]=1

#                 chiral_type = atom.GetChiralTag()
#                 if str(chiral_type) != 'CHI_UNSPECIFIED':
#                     chiral_type_idx = self.chiral_alphabet.index(str(chiral_type))
#                     X_chiral[ref_atom_idx, chiral_type_idx]=1
                
#                 is_aromatic = 1
#                 if not mol.GetAtomWithIdx(ref_atom_idx).GetIsAromatic():
#                     is_aromatic=0
#                 X_rest[ref_atom_idx,-3] = is_aromatic

#                 is_chiral = 1
#                 if chiral_type == "CHI_UNSPECIFIED" or chiral_type==0:
#                     is_chiral=0
#                 X_rest[ref_atom_idx,-2] = is_chiral


#                 atom_degree = len(atom.GetBonds()) #total number of bonds attached to the atom
#                 atom_degree_idx = self.degree_alphabet.index(atom_degree)
#                 X_degree[ref_atom_idx, atom_degree_idx] = 1

#                 nH = atom.GetNumImplicitHs()
#                 nH_idx = self.hydrogens_alphabet.index(nH)
#                 X_hydrogens[ref_atom_idx, nH_idx]=1

#                 atom_mass = atom.GetMass()/100
#                 X_rest[ref_atom_idx,-1] = atom_mass
#                 ####################--- ATOM FEATURES--###################

#                 ####################---oldFEATURES--###################
        
#                 #bond = mol.GetBondBetweenAtoms(ref_atom_idx,neighbor_atom_idx)
#                 #atom_mass = atom.GetMass()/100
#                 #bond_type = bond.GetBondType()
#                 #explicit_valence = mol.GetAtomWithIdx(ref_atom_idx).GetExplicitValence()
#                 #atom_mass = atom.GetMass()/100
                
#                 # for debug
#                 #print(symbol, formal_charge, radical_electrons, hybridization_type, chiral_type, is_chiral, atom_degree, nH)
                
#                 #X[ref_atom_idx, X_init_features] = atom_degree
#                 #X[ref_atom_idx, X_init_features+1] = self.bond_type_enc[bond_type]
#                 #X[ref_atom_idx, X_init_features+1] = nH
#                 #X[ref_atom_idx, X_init_features+2] = explicit_valence
#                 #if mol.GetAtomWithIdx(ref_atom_idx).GetIsAromatic():
#                 #    X[ref_atom_idx, X_init_features+2] = 1
#         X = np.hstack([X_elements, X_hybrdization, X_degree, X_hydrogens, X_chiral, X_rest])       
#         assert X.shape[1] == self.total_features 
#         if graph:
#             return nx.from_numpy_matrix(A)
        
#         return A, X, A_bond_type, A_conjugated, A_stero


# class ExpressionDataset(Data.Dataset):
#     def __init__(self, X, Y):
#         self.X = X
#         self.y = Y

#     def __len__(self):
#         return (len(self.X))

#     def __getitem__(self, i):
#         data = self.X[i]
#         data = torch.from_numpy(data)

#         y = torch.LongTensor([self.y[i]])
#         return (data.float(), y)

       
# class MorganDataset(Data.Dataset):
#     def __init__(self, inp, target):

#         self.inp = inp
#         self.y = target

#     def __len__(self):
#         return self.y.shape[0]

#     def __getitem__(self,i):

#         fp = self.inp.iloc[i].values   
# #         sm = self.inp.iloc[i].name
#         y = self.y.iloc[i].values

#         y = torch.from_numpy(y).type(torch.FloatTensor)

#         fp = torch.from_numpy(fp).type(torch.FloatTensor)
#         return fp, y

# class DAdataset(Data.Dataset):
#     def __init__(self, expr, disease, domain):
#         self.expr = expr
#         self.disease = disease
#         self.domain = domain
        
#         self.disease_indices = {disease:np.where(self.disease==disease) for disease in self.disease}
        
#     def __len__(self):
#         return self.expr.shape[0]
    
#     def __getitem__(self,i):
        
#         x1 = self.expr[i]
#         x2 = np.zeros(x1.shape)
        
#         y = self.disease[i]
#         sample_domain = self.domain[i]
        
#         z = np.zeros(sample_domain.shape, dtype=int).reshape(-1,1)
# #         print('z',z)
#         u = np.ones(sample_domain.shape).reshape(-1,1)
# #         print('u',u)
#         domain_positive_ids = np.where(self.domain==sample_domain)[0]
#         domain_negative_ids = np.where(self.domain!=sample_domain)[0]
            
#         positive_ids = np.setdiff1d(domain_positive_ids, self.disease_indices[y])
#         negative_ids = np.intersect1d(domain_negative_ids, self.disease_indices[y])
        
        
#         if np.random.random()>0.5 and len(positive_ids) >= 1:
# #             print('len(positive_ids)',len(positive_ids))
#             siamese_index = np.random.choice(positive_ids)
# #             print('siamese_index', siamese_index)
#             x2 = self.expr[siamese_index]
#             z[0] = 1
#         elif np.random.random()>0.5 and len(negative_ids) >= 1:
# #             print('len(negative_ids)',len(negative_ids))
#             siamese_index = np.random.choice(negative_ids)
# #             print('siamese_index', siamese_index)
#             x2 = self.expr[siamese_index]
#             z[0] = 0
#         else:
# #             print('u')
#             u[0] = 0
            
#         x1 = torch.FloatTensor(x1)
#         x2 = torch.FloatTensor(x2)
#         y = torch.LongTensor(y.reshape(-1,1))[0]
#         z = torch.LongTensor(z)[0]
#         u = torch.FloatTensor(u)[0]

#         return x1, x2, y, z, u

# class ExpressionResponseDataset(Data.Dataset):
#     def __init__(self, X, Y):
#         self.X = X # pandas dataframe of (samples, genes)
#         self.y = Y # pandas dataframe of (samples, response)


#     def __len__(self):
#         return self.y.shape[0]

#     def __getitem__(self, i):
#         #data = self.X.iloc[i]
#         #print('data',data)
#         #sample = self.X.index[i]
#         #sample = self.y.iloc[i].index
#         #print('sample',sample)
#         response = self.y.iloc[i].values # response vector across all drugs for the sample
#         sample = self.y.iloc[i].name
        
#         data = self.X.loc[sample]

#         data = torch.from_numpy(data.values).type(torch.FloatTensor)
#         y = torch.from_numpy(response).type(torch.FloatTensor)
#         #assert data.shape[0] == y.shape[0]

#         return (data, y)