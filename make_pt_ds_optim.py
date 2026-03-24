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

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

def get_xy_single_var(cms_path:str, ogs_path:str, cms2ogs_map:Dict[str, str], cms_name: str, split: str):
    '''
        Returns the input and target values corresponding to a given variable in the data folder.

        Args:
            data_path (str): path to the data directory
            cms2ogs_map: dictionary associating Copernicus Marine names to CADEAU names
            cms_name (str): variable name in the Copernicus Marine notation.
            split (str): type of dataset we are using (should be "train", "test" or "validation")
        Returns:
            x_arr (numpy array): input of the DL model for the given variable
            y_arr (numpy array): target of the DL model for the given variable
    '''
    ogs_name = cms2ogs_map[cms_name]
    x_list = []; y_list = []

    # OGS
    if ogs_path is not None:
            ogs_path = os.path.join(ogs_path, ogs_name)
            # natsort ordina in base al numero es. 5 1 10  -> sorted ordina in base alla stringa: 1 10 5 mentre natsort in base al significato numeriico: 1 5 10
            ogs_filenames = natsort.natsorted(os.listdir(ogs_path))
            with alive_bar(len(ogs_filenames), title=f"Processing OGS {split} data for {ogs_name}...") as bar:
                for filename in ogs_filenames:
                    file_path: str = os.path.join(ogs_path, filename)
                    with nc.Dataset(file_path) as file_ds: 
                        y_list.append(file_ds[ogs_name][:].data)
                    bar()

    # CMS
    cms_path = os.path.join(cms_path, cms_name)
    cms_filenames = natsort.natsorted(os.listdir(cms_path))
    with alive_bar(len(cms_filenames), title=f"Processing CMS {split} data for {cms_name}...") as bar:
        for filename in cms_filenames:
            file_path = os.path.join(cms_path, filename)
            with nc.Dataset(file_path) as file_ds:  
                x_list.append(file_ds[cms_name][:].data)
            bar()

    if ogs_path == None:
        return np.array(x_list), None
    else:
        return np.array(x_list), np.array(y_list)



# normalization with broadcasting
# def normalize(ds:np.array, means: np.array, stds:np.array, mask:np.array):
#     ds = ds[:, None, :, :, :] # Samples, Channel = None = 1, Depth 27, Height 300, Width 494
#     mask = np.repeat(mask, ds.shape[0], axis=0)
#    # print(ds.shape)
#    # print(mask.shape)
#     ds = np.ma.masked_array(ds, mask)

#     # creiamo shape dinamica
#     shape = [1, -1] + [1]*(ds.ndim-2)

#     ds = (ds - means.reshape(shape)) / stds.reshape(shape)
#     # print('mean shape:', means.shape)
#     # print('sd shape:', stds.shape)
#     ds = np.ma.masked_invalid(ds).filled(1e7)
#     # print(ds.shape)
#     return ds.astype(np.float32)

def normalize(ds: np.array, means: np.array, stds: np.array, mask: np.array):
    """
    Normalizza un dataset multi-canale con broadcasting.
    
    ds: np.array, shape = (samples, channels, depth, H, W) 
    means: np.array, shape = (channels,)
    stds: np.array, shape = (channels,)
    mask: np.array, shape = (depth, H, W) per singolo campione
    """
    # ripeti la mask lungo samples
    print('ds input shape:', ds.shape)
    ds = ds[:, None, :, :, :]  # aggiungi asse channel = 1
    print('ds shape:', ds.shape)
    mask = np.repeat(mask, ds.shape[0], axis=0)  
    ds = np.ma.masked_array(ds, mask)                      # shape = (samples, 1, depth?, H, W)
    print('mask shape:', mask.shape)
    print('ds shape after masking:', ds.shape)
    
    # broadcasting dei canali
    shape = [1, -1] + [1]*(ds.ndim-2)  # [1, channels, 1,1,1,...] per diff dimensioni spaziali
    ds = (ds - means.reshape(shape)) / stds.reshape(shape)
    
    ds = np.ma.masked_invalid(ds).filled(1e7)
    print('output shape', ds.shape)
    return ds.astype(np.float32)



# # normalization with for loop
# def normalize(ds:np.array, means:np.array, stds:np.array, mask:np.array):
#     """Compute normalization of the dataset, given its mean
#        and standard deviation.
#        MASK: array booleano della stessa forma spaziale di un singolo campione, che indica quali elementi devono essere ignorati nella normalizzazione.
#        Le posizioni mascherate vengono infine riempite con 1e7 per indicare valori “inutilizzabili”.
#     """
#     ds = ds[:, None, :, :, :] 
#     mask = np.repeat(mask, ds.shape[0], axis=0) # Qui la maschera originale, che copre solo le dimensioni spaziali di un singolo campione, viene ripetuta lungo l’asse dei campioni, in modo che corrisponda alla forma completa di ds.
#     ds = np.ma.masked_array(ds, mask) # crea un array mascherato, dove tutti gli elementi dove mask=True vengono ignorati nelle operazioni matematiche.
#     normalized_ds = np.zeros_like(ds, dtype=np.float32)
#     print('mask shape:', mask.shape)
#     print('ds shape:', ds.shape)


#     print(ds.ndim)
#     # NOTA: questi due cicli for probabilmente potrebbero essere sostituiti con qualocsa di più efficiente
#     for i in range(ds.shape[0]):  # Iterate over each data sample
#         for v in range(ds.shape[1]):  # Iterate over each channel
#             if ds.ndim == 4:  # Check if the image is 2D or 3D
#                 normalized_ds[i, v, :, :] = np.ma.masked_invalid((ds[i, v, :, :] - means[v]) / stds[v]).filled(1e7) # .masked_invalid maschera eventuali valori NaN o inf risultanti dalla divisione, sostituendoli con con 1e7
#             else:
#                 normalized_ds[i, v, :, :, :] = np.ma.masked_invalid((ds[i, v, :, :, :] - means[v]) / stds[v]).filled(1e7)

#     # # MODIFICA QUI PER DEBUG
#     # print("\n[DEBUG normalize] dtype:", ds.dtype)
#     # print("[DEBUG normalize] min/max:", ds.min(), ds.max())
#     # print("[DEBUG normalize] std min:", np.min(stds))
#     print('norms ds:', normalized_ds.shape)
#     return normalized_ds.data





def make_var_dataset(cms_path:str, ogs_path:str, save_path:str, stat_path:str, train_path:str, var: str, cms2ogs_map:Dict[str, str], split:str):
    '''
        Saves the Pytorch dataset corresponding to a given variable in the output folder.

        Args:
            data_path (str): path of the folder with data
            save_path (str): where we want to save the .pt file
            stat_path (str): inside the path must be a folder with statistics (mean, sd) already calculated on the train set
            train_path (str): path of folder with train data
            var (str): name of variable in the Copernicus Marine notation.
            cms2ogs_map: dictionary associating to Copernicus Marine names CADEAU names
            split (str): type of the dataset we are going to make in .pt format, should be: "train", "test" or "validation

    '''
    print("[INFO] Training set always downloaded to apply the mask")

    cms = []
    ogs = []

    x_means = []
    x_stds = []
    y_means = []
    y_stds = []

    # leggo 
    varx_list, vary_list = get_xy_single_var(cms_path, ogs_path, cms2ogs_map, var, split)
    cms.append(varx_list)
    ogs.append(vary_list)


    # prendo medie e sd per la normalizzazione 
    with open(f'{stat_path}/cms/stat_cms_{var}.txt', 'r') as file:
        line_elements = []
        for line in file:
            if line.strip():
                line_elements.append(np.float32(line))
    x_means.append(line_elements[0])
    x_stds.append(line_elements[1])

    with open(f'{stat_path}/ogs/stat_ogs_{var}.txt', 'r') as file:
        line_elements = []
        for line in file:
            if line.strip():
                line_elements.append(np.float32(line))
    y_means.append(line_elements[0])
    y_stds.append(line_elements[1])

    x_means = np.array(x_means)
    x_stds = np.array(x_stds)
    y_means = np.array(y_means)
    y_stds = np.array(y_stds)

    x = cms[0]
    y = ogs[0]

    
    # Calcola mask solo sul training set
    if split == 'train':
        mask = x[0] > 100000
    else:
        # Legge solo il CMS train path per calcolare la maschera
        varx_train, _ = get_xy_single_var(train_path, None, cms2ogs_map, var, 'train')
        mask = varx_train[0] > 100000
        # cms_train = []
        # varx_list, _ = get_xy_single_var(train_path, None, cms2ogs_map, var, split)
        # cms_train.append(varx_list)
        # x = np.array(cms_train)
        # mask = x[0] > 100000

    x = normalize(x, x_means, x_stds, mask)
    y= normalize(y, y_means, y_stds, mask)
    torch_ds = TensorDataset(torch.Tensor(x), torch.Tensor(y))

 
    os.makedirs(save_path, exist_ok=True)
    ds_save_path = os.path.join(save_path,f"{var}_{split}_dataset.pt")
    print(f"Saving the {split} pytorch datasets in {save_path}")
    torch.save(torch_ds, ds_save_path)





if __name__== "__main__":
    """Script to construct the training and the test dataset
    (after that test and training data have already been divided).
    Inputs:
        -dp (str): data path for training and test. 
        -op (str): output path, where we want to save the output


    """
    i = 1
    var = None
    save_path = None
    split = None
    conf_path = None

    while i < len(sys.argv):

        if sys.argv[i] == "-op":
            if save_path != None: raise ValueError("Repeated input for -op: output path")
            if i+1 >= len(sys.argv): raise ValueError("Missing path for output path")
            save_path = sys.argv[i+1]; i+= 2

        elif sys.argv[i] == "-conf":
            if conf_path: raise ValueError("Repeat path for configuration json file") 
            conf_path = sys.argv[i+1]; i += 2
        
        elif sys.argv[i] == "-v":
            if var: raise ValueError("Repeated input for variable")
            var = sys.argv[i+1]; i += 2

        elif sys.argv[i] == "-split":
            if split: raise ValueError('Repeated input for -split')
            split = sys.argv[i+1]; i += 2

       
        
    if split not in ["train", "test", "validation"]:
            raise ValueError("split must be train, test or validation")

    if not var:
        print("[WARNING] No variables provided with -v. No variable dataset will be created.")

    with open(conf_path, 'r') as f:
        conf = json.load(f)
    conf = obj(conf)

 
    with open(conf.map_path, 'r') as f:
        cms2ogs_map = json.load(f)

    print(f"[make_dataset for variables {var}] Starting execution")

    cms_train_path = conf.cms_train_path

    if var is not None:
        if split == 'train':
            make_var_dataset(conf.cms_train_path,conf.ogs_train_path, save_path, conf.stat_path, cms_train_path, var, cms2ogs_map, split)
        elif split == 'test':
            make_var_dataset(conf.cms_test_path,conf.ogs_test_path, save_path, conf.stat_path, cms_train_path, var, cms2ogs_map, split)
        else:
            make_var_dataset(conf.cms_val_path,conf.ogs_val_path, save_path, conf.stat_path, cms_train_path, var, cms2ogs_map, split)


    
