import netCDF4 as nc
import torch
import numpy as np
import os
import sys
from torch.utils.data import TensorDataset
from alive_progress import alive_bar
from typing import List, Dict
import json
import natsort
import matplotlib.pyplot as plt



class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

def get_river_vector(data_path:str, split:str):
    """
    """
    x_list = []

    cms_filenames = sorted(os.listdir(data_path))

    with alive_bar(len(cms_filenames), title=f"Processing river {split} data ...") as bar:
        for filename in cms_filenames:
            file_path = os.path.join(data_path, filename)
            x=np.loadtxt(file_path)
            x_list.append(x)
            bar()
    return np.array(x_list)


def make_rivers_dataset(data_path:str, save_path:str, split:str, altro = None):
    """Construct river training and test torch dataset.
    """
    os.makedirs(save_path, exist_ok=True)
    suffix = f"_{altro}" if altro else ""
    ds_save_path = os.path.join(save_path, f"rivers_{split}{suffix}.pt")
    rivers = get_river_vector(data_path, split)
    print(f"Saving the pytorch river {split} datasets in {save_path}")
    torch.save(torch.Tensor(rivers), ds_save_path)
    print("Saved!")


# CREA FUNZIONE CHE CONTROLLA IL CONF

if __name__== "__main__":
    """Script to construct the training and the test dataset
    (after that test and training data have already been divided).
    Inputs:
        -dp (str): data path for training and test. 
        -op (str): output path, where we want to save the output


    """
    i = 1
    save_path = None
    split = None
    conf_path = None
    altro = None

    while i < len(sys.argv):

        if sys.argv[i] == "-op":
            if save_path != None: raise ValueError("Repeated input for -op: output path")
            if i+1 >= len(sys.argv): raise ValueError("Missing path for output path")
            save_path = sys.argv[i+1]; i+= 2

        elif sys.argv[i] == "-conf":
            if conf_path: raise ValueError("Repeat path for configuration json file") 
            conf_path = sys.argv[i+1]; i += 2

        elif sys.argv[i] == "-split":
            if split: raise ValueError('Repeated input for -split')
            split = sys.argv[i+1]; i += 2

        elif sys.argv[i] == "-altro":
            if i+1 < len(sys.argv) and not sys.argv[i+1].startswith("-"):
                altro = sys.argv[i+1]
                i += 2
            else:
                # niente valore → ignora e lascia None
                i += 1

        else:
            raise ValueError(f"Unknown argument: {sys.argv[i]}")


       
    # controllo sullo split   
    if split not in ["train", "test", "validation"]:
            raise ValueError("split must be train, test or validation")

    # lettura config
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    conf = obj(conf)


# CHIAMA FUNZIOME CHE CONTROLLA IL CONF 


    # chiamiamo funzione che crea i file .pt 

    if split == 'train':
        make_rivers_dataset(conf.rivers_train_path, save_path, split, altro)
    elif split == 'test':
        make_rivers_dataset(conf.rivers_test_path, save_path, split, altro)
    else:
        make_rivers_dataset(conf.rivers_val_path, save_path, split, altro)

    