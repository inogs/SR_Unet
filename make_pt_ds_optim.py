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



def plot_boxplot_raw_input_target(save_path, varx_list: np.array, vary_list: np.array, mask: np.array, var_list: list, split: str):
    """
    Disegna un boxplot dei valori grezzi non mascherati dei dataset input (CMS) e target (OGS)
    varx_list: array grezzo CMS, shape = (samples, channels, H, W, D)
    vary_list: array grezzo OGS, shape = (samples, channels, H, W, D)
    mask: mask originale usata per i dati
    var_list: lista delle variabili considerate
    split: train/test/validation
    """
    # espandi la mask sui samples
    mask_fullx = np.repeat(mask, varx_list.shape[0], axis=0)
  
    mask_fully = np.repeat(mask, vary_list.shape[0], axis=0)
    
    x_masked = np.ma.masked_array(varx_list, mask_fullx)
    y_masked = np.ma.masked_array(vary_list, mask_fully) 

    x_vals = x_masked.compressed()
    y_vals = y_masked.compressed() 
    plt.figure(figsize=(10,6))
    # plt.boxplot([np.log1p(x_vals), np.log1p(y_vals)], tick_labels=["Input (CMS)", "Target (OGS)"]) 
    plt.boxplot([x_vals, y_vals], tick_labels=["Input (CMS)", "Target (OGS)"]) # np.log1p



    plt.title(f"Boxplot valori non normalizzati solo quelli non mascherati - {split} - variabili: {', '.join(var_list)}")
    plt.ylabel("Valori non normalizzati")
    save_path = os.path.join(save_path, f"boxplot_{split}{var_list[0]}.png")
    plt.savefig(save_path)
    print(f"[INFO] Boxplot salvato in {save_path}")


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




# normalization with for loop -> MANTIENI PER ORA QUESTA FUNZIONE PERCHE' SAI CHE E' CORRETTA ED E' CONCETTULMENTE PIU' SEMPLICE quindi all'nizio va bene
def normalize(ds:np.array, means:np.array, stds:np.array, mask:np.array):
    """Compute normalization of the dataset, given its mean
       and standard deviation.
       MASK: array booleano della stessa forma spaziale di un singolo campione, che indica quali elementi devono essere ignorati nella normalizzazione.
       Le posizioni mascherate vengono infine riempite con 1e7 per indicare valori “inutilizzabili”.
    """
    print('input mask shape', mask.shape) # dovrebbe essere (1 = C, 27, 300, 494)
    mask = np.repeat(mask, ds.shape[0], axis=0) # Qui la maschera originale, che copre solo le dimensioni spaziali di un singolo campione, viene ripetuta lungo l’asse dei campioni, in modo che corrisponda alla forma completa di ds.
    ds = np.ma.masked_array(ds, mask) # crea un array mascherato, dove tutti gli elementi dove mask=True vengono ignorati nelle operazioni matematiche.
    print("[DEBUG not normalized] min/max:", np.min(np.ma.masked_array(ds, mask)), np.max(np.ma.masked_array(ds, mask)))

    normalized_ds = np.zeros_like(ds, dtype=np.float32) # creazione di un 'contenitore' vuoto dove metterci i dati normalizzati
    # print('mask shape2:', mask.shape)
    # print('ds shape:', ds.shape)
    # print(ds.ndim)
    # NOTA: questi due cicli for probabilmente potrebbero essere sostituiti con qualocsa di più efficiente -> BROADCASTING ?
    for i in range(ds.shape[0]):  # Iterate over each data sample
        print("[DEBUG not normalized] min/max:", np.min(np.ma.masked_array(ds, mask)), np.max(np.ma.masked_array(ds, mask)))

        for v in range(ds.shape[1]):  # Iterate over each channel
           
            # aggiungi where per ignorare maschere --> prova sia mask che ~mask !!!
           normalized_ds[i, v, :, :, :] = np.ma.masked_invalid(np.ma.divide((ds[i, v, :, :, :] - means[v]), stds[v])).filled(1e7) # maschera eventuali valori nan o inf e sostituisce i valori mascherati con 1e7

    # print('output shape:', normalized_ds.shape)
    # print("\n[DEBUG masked array] dtype:", ds.dtype)
    # print("[DEBUG masked array] min/max:", np.min(ds), np.max(ds))
    # print("[DEBUG] std, mean", stds, means)
    
    # print("\n[DEBUG normalized ds] dtype:", normalized_ds.dtype)
    # print("[DEBUG normalized ds] min/max:", np.min(normalized_ds), np.max(normalized_ds))

    # print("[DEBUG normalized ds mascherato] min/max:", np.min(np.ma.masked_array(normalized_ds, mask)), np.max(np.ma.masked_array(normalized_ds, mask)))

    return normalized_ds.data



# NORMALIZE CHE NON DA' ERRORE DI OVERFLOW
# def normalize(ds: np.array, means: np.array, stds: np.array, mask: np.array):
#     print()
#     # ripeti la mask lungo i campioni
#     mask_full = np.repeat(mask, ds.shape[0], axis=0)
    
#     # crea array mascherato
#     ds_masked = np.ma.masked_array(ds, mask_full)
    
#     # prepara contenitore finale
#     normalized_ds = np.zeros_like(ds, dtype=np.float32)
    
#     for i in range(10):
#         for v in range(ds.shape[1]):
#             # operazioni solo sui valori non mascherati
#             valid_data = ds_masked[i, v, :, :, :].compressed()  # prendi solo valori non mascherati
#             if valid_data.size > 0:
#                 normalized_values = (valid_data - means[v]) / stds[v]
#             else:
#                 print('No valid data')
            
#             # ricostruisci l'array completo con 1e7 per i mascherati
#             temp = np.full(ds_masked[i, v, :, :, :].shape, 1e7, dtype=np.float32)
#             temp[~ds_masked[i, v, :, :, :].mask] = normalized_values
#             normalized_ds[i, v, :, :, :] = temp

#         print(i)
#         print("[DEBUG normalized ds con maschera] min/max:", np.min(normalized_ds), np.max(normalized_ds))
#         print("[DEBUG normalized ds solo valori mascherati] min/max:", np.ma.masked_array(normalized_ds, mask_full).max(), np.ma.masked_array(normalized_ds, mask_full).min())
#         print("[DEBUG normalized ds solo valori mascherati] min/max:", ds_masked.data.max(), ds_masked.data.min())
#         print()

#     return normalized_ds




# DEBUGGING ERRORE OVERFLOW
# def normalize(ds:np.array, means:np.array, stds:np.array, mask:np.array):
#     mask = np.repeat(mask, ds.shape[0], axis=0)
#     ds = np.ma.masked_array(ds, mask) 
#     normalized_ds = np.zeros_like(ds, dtype=np.float32)
#     for i in range(10):  
#         for v in range(ds.shape[1]):
#             print('sottrazione')
#             tmp = ds[i, v, :, :, :] - means[v]

#             print('divisione sd')
#             print("dtype:", tmp.dtype)

#             raw = tmp.data
#             print("REAL max:", raw.max())
#             print("REAL min:", raw.min())

#             with np.errstate(over='raise', divide='raise', invalid='raise'):
#                 try:
#                     tmp2 = tmp / stds[v]
#                     print('sd:', stds[v])

#                 except FloatingPointError:
#                     print("!!!!  ERRORE divisione !!!!")
#                     print("i:", i, "v:", v)
#                     print('sd:', stds[v])
#                     print("REAL max:", raw.max())
#                     print("REAL min:", raw.min())

#                     # continua comunque il loop
#                     tmp2 = np.full_like(tmp, 1e7)  # oppure np.nan

#             print('invalid e filled')

#             normalized_ds[i, v, :, :, :] = np.ma.masked_invalid(tmp2).filled(1e7)
#     return normalized_ds.data


# def normalize(ds: np.array, means: np.array, stds: np.array, mask: np.array):
#     """
#     Normalizza un dataset multi-canale con broadcasting.
    
#     ds: np.array, shape = (samples, channels, depth, H, W) 
#     means: np.array, shape = (channels,)
#     stds: np.array, shape = (channels,)
#     mask: np.array, shape = (depth, H, W) per singolo campione
#     """
#     # ripeti la mask lungo samples
#     print('ds input shape:', ds.shape)
#     ds = ds[:, :, :, :]  # aggiungi asse channel = 1
#     print('ds shape:', ds.shape)
#     mask = np.repeat(mask, ds.shape[0], axis=0)  
#     ds = np.ma.masked_array(ds, mask)                      # shape = (samples, 1, depth?, H, W)
#     print('mask shape:', mask.shape)
#     print('ds shape after masking:', ds.shape)
    
#     # broadcasting dei canali
#     shape = [1, -1] + [1]*(ds.ndim-2)  # [1, channels, 1,1,1,...] per diff dimensioni spaziali
#     ds = (ds - means.reshape(shape)) / stds.reshape(shape)
    
#     ds = np.ma.masked_invalid(ds).filled(1e7)
#     print('output shape', ds.shape)
#     return ds.astype(np.float32)



def make_var_dataset(cms_path:str, ogs_path:str, save_path:str, stat_path:str, train_path:str, var_list: List[str], cms2ogs_map:Dict[str, str], split:str, altro: str = None): 
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

    for var in var_list:
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

    x = np.array(list(zip(*cms)))
    y = np.array(list(zip(*ogs)))



    
    # Calcola mask solo sul training set
    if split == 'train':
        # print('x shape:', x.shape)
        mask = x[0] > 100000 # np.any(x > 100000, axis=0)
    else:
        # Legge solo il CMS train path per calcolare la maschera 
        # GIUSTO SE LE MASCHERE SONO UGUALI per tutti i timestamp !
        # qui aggiungo un for loop altrimenti l'output mashera mi dava shape sbagliata, cioè senza la shape channel, dato che per tutte le variabili CREDO ci siano le stesse maschere
        # invece di iterare per le variabili si potrebbe fare un reshape della maschera e basta 
        
        # OPZIONE 1. LOOP PER LE VARIABILI
        # for var in var_list:
        #     varx_train, _ = get_xy_single_var(train_path, None, cms2ogs_map, var, 'train') 
        #     varx.append(varx_train)
        # varx  = np.array(list(zip(*varx)))  # print('varx', varx.shape) dovrebbe essere (717, num channel, etc)
        # mask = varx[0] > 100000 # np.any(x > 100000, axis=0) se vogliamo considerare tutti i timestamp

        # OPZIONE 2. RESHAPE 
        # se invece le mask sono uguali per tutte le variabili possiamo mettere noi la shape del channel senza iterare per tutte le variabili -> CONFERMO SONO uguali per ogni variabile ed ogni timestamp! 
        varx_train, _ = get_xy_single_var(train_path, None, cms2ogs_map, var_list[0], 'train')
        varx_train = varx_train[:, None, ...]  # (batch, channel, H, W, D)
        mask = varx_train[0] > 100000  # (batch, H, W, D)
    
    # plot_boxplot_raw_input_target(save_path, x, y, mask, var_list, split)
    
    x = normalize(x, x_means, x_stds, mask)
    y= normalize(y, y_means, y_stds, mask)
    torch_ds = TensorDataset(torch.Tensor(x), torch.Tensor(y))


 
    if len(var_list) == len(cms2ogs_map):
        name = "all_"
    else:
        name = ""
        for var in var_list:
            name = name + f"{var}_"

    os.makedirs(save_path, exist_ok=True)
    ds_save_path = os.path.join(save_path,f"{name}{split}_dataset{altro}.pt")
    print(f"Saving the {split} pytorch datasets in {save_path}")
    torch.save(torch_ds, ds_save_path)



# CREA FUNZIONE CHE CONTROLLA IL CONF

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
    var_list = []
    altro = None

    while i < len(sys.argv):

        if sys.argv[i] == "-op":
            if save_path != None: raise ValueError("Repeated input for -op: output path")
            if i+1 >= len(sys.argv): raise ValueError("Missing path for output path")
            save_path = sys.argv[i+1]; i+= 2

        elif sys.argv[i] == "-conf":
            if conf_path: raise ValueError("Repeat path for configuration json file") 
            conf_path = sys.argv[i+1]; i += 2
        
        elif sys.argv[i] == "-v":
            if var_list != []: raise ValueError("Repeated input for variable")
            if sys.argv[i+1] == "all":
                var_list = "all"
                i+= 2
            else:
                while i < len(sys.argv)-1:
                    if not sys.argv[i+1].startswith('-'):
                        var_list.append(sys.argv[i+1]) ; i+= 1
                    else:
                        break
                i+= 1

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


       
    # controllo sullo split   
    if split not in ["train", "test", "validation"]:
            raise ValueError("split must be train, test or validation")

    # lettura config
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    conf = obj(conf)

    # lettura dizionario cms <-> ogs 
    with open(conf.map_path, 'r') as f:
        cms2ogs_map = json.load(f)

    # opzione 'considera tutte le variabili'
    if var_list == "all":
        var_list = list(cms2ogs_map.keys())  

# CHIAMA FUNZIOME CHE CONTROLLA IL CONF 

    # controllo su var list
    if var_list == []:
        print("[WARNING] No variables provided with -v. No variable dataset will be created.")
    else:
        print(f"[make_dataset for variables {var_list}] Starting execution")

    # salviamo il path del train
    cms_train_path = conf.cms_train_path

    # chiamiamo funzione che crea i file .pt 
    if var_list != []:
        if split == 'train':
            make_var_dataset(conf.cms_train_path,conf.ogs_train_path, save_path, conf.stat_path, cms_train_path, var_list, cms2ogs_map, split, altro)
        elif split == 'test':
            make_var_dataset(conf.cms_test_path,conf.ogs_test_path, save_path, conf.stat_path, cms_train_path, var_list, cms2ogs_map, split, altro )
        else:
            make_var_dataset(conf.cms_val_path,conf.ogs_val_path, save_path, conf.stat_path, cms_train_path, var_list, cms2ogs_map, split, altro)

    