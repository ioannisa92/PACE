import os
import setuptools

DATADIR='./data/'
RESDIR="./results/"
MODELDIR="./saved_models/"
PLOTDIR='./plots/'
PRETRAINEDMODELS='./pretrained_models'

if not os.path.exists(RESDIR):
    os.makedirs(RESDIR)
    
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)
    
if not os.path.exists(MODELDIR):
    os.makedirs(MODELDIR)

if not os.path.exists(PLOTDIR):
    os.makedirs(PLOTDIR)

if not os.path.exists(PRETRAINEDMODELS):
    os.makedirs(PRETRAINEDMODELS)
    
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


required = ['numpy==1.16.1',
            'sklearn',
            'scipy==1.3.2',
            'sklearn',
            'scipy==1.3.2',
            'tqdm==4.38.0',
            'matplotlib==3.0.3',
            'pubchempy==1.0.4',
            'networkx==2.2',
            'pandas==0.24.2']


setuptools.setup(
    name="Drug response prediction with Patient Adaption and Chemical Embedding (PACE)",
    version="0.0.1",
    author="Ioannis Anastopoulos",
    author_email="ianastop@ucsc.edu",
    description="Repository for training PACE and loading pretrained model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ioannisa92/PACE",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)