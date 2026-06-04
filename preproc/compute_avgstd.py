import os
import sys
import json
import netCDF4 as nc
import numpy as np
import numpy.ma as ma

def get_avg_std(var:str, data_type:str, data_path:str, cms2ogs_map:dict[str, str]):
    ''' Get mean and std for variable var.
        Args:
            var: variable of interest
            data_type: either cms or ogs
            data_path: path of the data files
            cms2ogs_map: json file with name conversion
    '''
    if data_type == "cms":
        path = os.path.join(data_path, "iCMS_nc", var) # SOLO TRAIN COPERNICUS nel momento in cui ho già splittao train e validation (train senza validation ha sempre il nome iCMS_nc)
    else:
        path = os.path.join(data_path, "NARF_nc", cms2ogs_map[var]) # SOLO TRAIN OGS (cadeau) (stesso discorso di sopra)

    # if data_type == "cms":
    #     path = os.path.join(data_path, "nc_iCMS_test", var) # SOLO TRAIN COPERNICUS nel momento in cui ho già splittao train e validation (train senza validation ha sempre il nome iCMS_nc)
    # else:
    #     path = os.path.join(data_path, "nc_OGS_test", cms2ogs_map[var]) # SOLO TRAIN OGS (cadeau) (stesso discorso di sopra)

    l = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        file_ds = nc.Dataset(file_path)
        if data_type == "cms":
            l.append(file_ds[var][:])
        else:
            l.append(file_ds[cms2ogs_map[var]][:])

    l1 = ma.concatenate(l)
    print(f"[INFO] Shape dopo concatenazione: {l1.shape}")
    print(f"[INFO] Numero totale elementi: {l1.size}")
    
    
    # Aggiunta stampa min e max
    # print(path)
    # print(f"[INFO] Minimo su tutto il volume: {l1.min()}")
    # print(f"[INFO] Massimo su tutto il volume: {l1.max()}")


    avg = np.average(l1)
    std = np.std(l1)
    print("avg", avg)
    print("std", std)
    return avg, std

if __name__ == '__main__':
    '''To be used in the preprocessing phase; for each variable in the json
    compute average and standard deviation and save them into a file.
    In this way, we don't need to compute again when we need to go back
    to starting values.
    '''
    i = 1
    data_path_input = None
    data_path_output = None
    map_path = None # dir del dizionario che mappa nome variabili csm e ogs

    while i < len(sys.argv):
        if sys.argv[i] == "-in":
            if data_path_input is not None:
                raise ValueError("Repeated input for data path")
            data_path_input = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-out":
            if data_path_output is not None:
                raise ValueError("Repeated input for data path")
            data_path_output = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-map":
            if map_path is not None:
                raise ValueError("Repeated path for map")
            map_path = sys.argv[i+1]
            i += 2
        else:
            i += 1

    if data_path_input is None:
        raise TypeError("Missing value for input data path")
    if data_path_output is None:
        raise TypeError("Missing value for output data path")


    # map_path = os.path.join(map_var_path, "cms2ogs.json")
    stat_path = os.path.join(data_path_output, "statistics")

    # Dopo aver ottenuto map_path da sys.argv
    if os.path.isdir(map_path):
        # Se è una cartella, aggiungi il nome del file
        map_path = os.path.join(map_path, "cms2ogs.json")

    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"Il file JSON della mappa non esiste: {map_path}")

    with open(map_path, 'r') as f:
        cms2ogs_map = json.load(f)

    var_list = list(cms2ogs_map.keys())

    for var in var_list:
        avl, stl = get_avg_std(var, 'cms', data_path_input, cms2ogs_map)
        path_cms = os.path.join(stat_path, "cms")
        if not os.path.exists(path_cms):
            os.makedirs(path_cms)
        with open(os.path.join(path_cms, f"stat_cms_{var}.txt"), "w") as file:
            file.write(f"{avl}\n{stl}\n")

        avl, stl = get_avg_std(var, 'ogs', data_path_input, cms2ogs_map)
        path_ogs = os.path.join(stat_path, "ogs")
        if not os.path.exists(path_ogs):
            os.makedirs(path_ogs)
        with open(os.path.join(path_ogs, f"stat_ogs_{var}.txt"), "w") as file:
            file.write(f"{avl}\n{stl}\n")
