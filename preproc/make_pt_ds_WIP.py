import netCDF4 as nc
import torch
import numpy as np
import os
import sys

# define main


if __name__== "__main__":
    # section, input paths
    path_target = "/home/giacomo/Desktop/Work_Folders/UNetDataAnalysis/Chla.test.txt"
    path_input = "/home/giacomo/Desktop/Work_Folders/UNetDataAnalysis/chl.test.txt"
    stat_taget = "/home/giacomo/Desktop/Work_Folders/UNetDataAnalysis/stat.Chla.test.txt"

    var_target = "Chla"
    var_input = "chl"

    # section, load file lists and check that they have the same length
    with open(path_target, 'r') as f:
        files_list_targets = [line.strip() for line in f if line.strip()]
    with open(path_input, 'r') as f:
        files_list_inputs = [line.strip() for line in f if line.strip()]
    if len(files_list_targets) != len(files_list_inputs):
        print("Error: the two lists have different lengths")
        sys.exit(1)
    print(f"    Number of files in target list: {len(files_list_targets)}")
    print(f"    Number of files in input list: {len(files_list_inputs)}")
    n_targets = len(files_list_targets)

    # section, load semples and define the dataset container
    ds_1 = nc.Dataset(files_list_targets[0])[var_target]
    print(f"    Shape of the variable {var_target} in the first file: {ds_1.shape}")
    storage_target = np.zeros((n_targets, 1, ds_1.shape[0], ds_1.shape[1], ds_1.shape[2]), dtype=ds_1.dtype)

    # i need a deep copy, if i close it the mask will be lost, but i need to close it to free the memory
    mask_ds_1 = np.ma.getmaskarray(ds_1)
    # ds_1.close()
    # mask_ds_1 = np.ma.getmaskarray(ds_1)

    print(f"    Container shape", storage_target.shape)

    # section, load statistics and print them, first one is mean, second one is std
    with open(stat_taget, 'r') as f:
        stats = [line.strip() for line in f if line.strip()]
    mean_target = float(stats[0])
    std_target = float(stats[1])
    print(f"    Mean of target variable: {mean_target}")
    print(f"    Std of target variable: {std_target}")

    # loop over the target files and store the data in the numpy array
    for i, file in enumerate(files_list_targets):
        ds = nc.Dataset(file)[var_target]
        # extract mask and check that it is the same as the one we have stored, if not print an error message and exit
        if not np.array_equal(mask_ds_1, np.ma.getmaskarray(ds)):
            print(f"Error: the mask of the file {file} is different from the one of the first file")
            sys.exit(1) 
        print(f"    Loaded file {i+1}/{n_targets}: {file}, mask check passed")
        # standardize ussing mean and std and the mask, if the mask is True, set the value to 0, otherwise standardize it
        storage_target[i, 0, :, :, :] = np.where(mask_ds_1, 0, (ds[:].data - mean_target) / std_target)
        # storage_target[i, 0, :, :, :] = ds[:].data
        # ds.close()
    # end of loop

    storage_input = np.zeros(storage_target.shape, dtype=storage_target.dtype)

    for i, file in enumerate(files_list_inputs):
        ds = nc.Dataset(file)[var_input]
        # extract mask and check that it is the same as the one we have stored, if not print an error message and exit
        if not np.array_equal(mask_ds_1, np.ma.getmaskarray(ds)):
            print(f"Error: the mask of the file {file} is different from the one of the first file")
            sys.exit(1) 
        print(f"    Loaded file {i+1}/{n_targets}: {file}, mask check passed")
        storage_target[i, 0, :, :, :] = np.where(mask_ds_1, 0, (ds[:].data - mean_target) / std_target)
        # ds.close()
    # end of loop

    print(f"    All files loaded, saving to pytorch format")
    torch_ds = torch.utils.data.TensorDataset(torch.Tensor(storage_target), torch.Tensor(storage_input))
    print(f"    Dataset shape: {torch_ds.tensors[0].shape}, {torch_ds.tensors[1].shape}")
    torch.save(torch_ds, "test_dataset.pt")
    print(f"    Dataset saved in test_dataset.pt")

    # load dataset and check that it is the same as the one we have saved
    loaded_ds = torch.load("test_dataset.pt")
    print(f"    Loaded dataset shape: {loaded_ds.tensors[0].shape}, {loaded_ds.tensors[1].shape}")
    if not torch.equal(torch_ds.tensors[0], loaded_ds.tensors[0]) or not torch.equal(torch_ds.tensors[1], loaded_ds.tensors[1]):
        print(f"Error: the loaded dataset is different from the one we have saved")
        sys.exit(1)
    print(f"    Dataset loaded successfully, check passed")


