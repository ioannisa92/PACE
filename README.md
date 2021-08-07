# Drug response prediction with Patient Adaption and Chemical Embedding (PACE) 
Pre-print: Pending


## Introduction
We develop a dual convergence model that combines expression information along with drug chemical information for drug response prediction. During training PACE adapts cell line expression encoding to that of a patient expression vector of the same tissue of origin. 
We show that such an adaptation combined with learning of the drug's structure leads to superior results in predicting patient drug response, while maintaining competitive performance in the cell line space. Our model is implemented in pytorch. We use torch_geometric to learn chemical graphs on drugs.

## Installation
Our model runs with python3. We recommend to use a recent version of python3 (eg. python>=3.6). \
We recommend using conda to create a virtual environment. \
Follow the steps bellow to install PACE:

```
conda create -n pace python=3.6
conda activate pace
git clone https://github.com/ioannisa92/PACE.git
python setup.py install
bash install_env.sh
```

Download archive with preprocessed data at: [data_link](https://drive.google.com/file/d/1mn3bageqCs-CZrIBbfCKHBWbmb0w2-ui/view?usp=sharing)

Unpack toy data to the data directory using this command in `data`:

```
tar -xvf pace_data.tar.gz
ccle_cdi.tsv --> cell line expression, drug, IC50 table
ccle_expr.tsv --> cell line expression (log2TPM)
ccle_labels.tsv --> cell line disease label used for adaptation
tcga_expr.tsv --> tcga expression (log2TPM)
tcga_labels.tsv --> tcga disease labels
```

Download archive with pretrained PACE models: [models_link](https://drive.google.com/file/d/1baAVfX5DOK3-8F_G2noX2MAPj56jcQrJ/view?usp=sharing)
```
tar -xvf pace_models.tar.gz
```
Unpack pretrained models using this command in `pretrained_models`:


Batch the Clinical Drug Response (CDR) data used in the paper for evaluation:
```
python batch_cdr.py -expr ./data/cdr_expr.tsv  -dir_out ./data/cdr/ -drug_fp GCN
```

Batch the any data to predict response on the desired drugs. This script required only the name of the drug. The structure and features of the drug are then extracted:
```
python batch_test_data.py -expr ./data/tcga_expr.tsv -dir_out ./data/tcga_test/ -drugs tamoxifen dabrafenib
```

Batch data to train PACE on. `-num_workers` refers to the number of CPUs used to batch the data:
```
python batch_train_data.py  -cl_expr ./data/ccle_expr.tsv  -tcga_expr ./data/tcga_expr.tsv -cl_labels ./data/ccle_labels.tsv -tcga_labels ./data/tcga_labels.tsv -cdi ./data/ccle_cdi.tsv -dir_out ./data/train/ -drug_fp GCN -num_workers 10
```

Train PACE:
```
python train.py -train_dir ./data/train/ -model_dir ./saved_models/
Training PACE...
Training model repeat0 on device cpu...
[Epoch 1] mmd_loss: 0.02196 | response_loss: 8.95425 | total_loss: 8.95447 | Time: 1.827 mins
Training model repeat1 on device cpu...
[Epoch 1] mmd_loss: 0.01793 | response_loss: 6.81983 | total_loss: 6.82001 | Time: 1.819 mins
Training model repeat2 on device cpu...
[Epoch 1] mmd_loss: 0.02015 | response_loss: 8.26018 | total_loss: 8.26038 | Time: 1.841 mins
Training model repeat3 on device cpu...
[Epoch 1] mmd_loss: 0.01746 | response_loss: 7.07806 | total_loss: 7.07823 | Time: 1.814 mins
Training model repeat4 on device cpu...
[Epoch 1] mmd_loss: 0.02212 | response_loss: 11.78739 | total_loss: 11.78761 | Time: 1.800 mins
Training model repeat5 on device cpu...
[Epoch 1] mmd_loss: 0.01769 | response_loss: 7.08442 | total_loss: 7.08460 | Time: 1.802 mins
Training model repeat6 on device cpu...
[Epoch 1] mmd_loss: 0.01953 | response_loss: 8.37816 | total_loss: 8.37836 | Time: 1.839 mins
Training model repeat7 on device cpu...
[Epoch 1] mmd_loss: 0.01867 | response_loss: 7.52785 | total_loss: 7.52804 | Time: 1.815 mins
Training model repeat8 on device cpu...
[Epoch 1] mmd_loss: 0.02199 | response_loss: 9.44609 | total_loss: 9.44631 | Time: 1.815 mins
Training model repeat9 on device cpu...
```

Predict on the CDR data using the pretrained PACE model. Models that have been trained with different data can also be loaded. Simply change the path of the trained models in `-model_dir`
```
python predict_cdr.py -i ./data/cdr/ -o cdr_test -model_dir pretrained_models
```

Predict drug response on any test data:
```
python predict.py -i ./data/tcga_test/ -o tcga_test -model_dir saved_models
```

