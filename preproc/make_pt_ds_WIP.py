import netCDF4 as nc
import torch
import numpy as np
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

n_workers = 24


def load_slice(index, file_path, variable, mean_value, std_value):
    with nc.Dataset(file_path) as dataset:
        array = dataset[variable][:]
        array_mask = np.ma.getmaskarray(array)
        data_slice = np.where(array_mask, 0, (array.data - mean_value) / std_value)

    return index, data_slice


def fill_storage_parallel(storage, files, variable, mean_value, std_value, label):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i, data_slice in executor.map(
            load_slice,
            range(len(files)),
            files,
            repeat(variable),
            repeat(mean_value),
            repeat(std_value),
            chunksize=1,
        ):
            print(f"    Loaded file {i+1}/{len(files)}: {files[i]}, {label} mask applied")
            storage[i, 0, :, :, :] = data_slice


if __name__== "__main__":
    # section, input paths
    path_target = "/leonardo_scratch/large/userexternal/gzuccari/NARF_nc/split.70.20.10.dir/Chla.val.txt"
    path_input = "/leonardo_scratch/large/userexternal/gzuccari/iCMS_nc/split.70.20.10.dir/chl.val.txt"
    stat_target = "/leonardo_scratch/large/userexternal/gzuccari/NARF_nc/split.70.20.10.dir/stat.Chla.val.txt"
    stat_input = "/leonardo_scratch/large/userexternal/gzuccari/iCMS_nc/split.70.20.10.dir/stat.chl.val.txt"

    var_target = "Chla"
    var_input = "chl"

    output_file_name = "chl_Chla_val_dataset.pt"

    # check that the four paths exist, if not print an error message and exit
    for path in [path_target, path_input, stat_target, stat_input]:
        if not os.path.exists(path):
            print(f"Error: the path {path} does not exist")
            sys.exit(1)
        else:
            print(f"    Path {path} exists, check passed")

    # section, load file lists and check that they have the same length

    # time this section
    start_time = time.time()
    with open(path_target, 'r') as f:
        files_list_targets = [line.strip() for line in f if line.strip()]
    with open(path_input, 'r') as f:
        files_list_inputs = [line.strip() for line in f if line.strip()]
    if len(files_list_targets) != len(files_list_inputs):
        print("Error: the two lists have different lengths")
        sys.exit(1)
    end_time = time.time()
    print(f"    Time taken to load file lists: {end_time - start_time} seconds")
    print(f"    Number of files in target list: {len(files_list_targets)}")
    print(f"    Number of files in input list: {len(files_list_inputs)}")
    n_targets = len(files_list_targets)

    # section, load semples and define the dataset container
    with nc.Dataset(files_list_targets[0]) as dataset_1:
        ds_1 = dataset_1[var_target]
        print(f"    Shape of the variable {var_target} in the first file: {ds_1.shape}")
        storage_target = np.zeros((n_targets, 1, ds_1.shape[0], ds_1.shape[1], ds_1.shape[2]), dtype=ds_1.dtype)

    print(f"    Container shape", storage_target.shape)

    # section, load statistics and print them, first one is mean, second one is std
    with open(stat_target, 'r') as f:
        stats = [line.strip() for line in f if line.strip()]
    mean_target = float(stats[0])
    std_target = float(stats[1])
    print(f"    Mean of target variable: {mean_target}")
    print(f"    Std of target variable: {std_target}")

    # read statistics for input variable, but we will not use them, just print them
    with open(stat_input, 'r') as f:
        stats = [line.strip() for line in f if line.strip()]
    mean_input = float(stats[0])
    std_input = float(stats[1])
    print(f"    Mean of input variable: {mean_input}")

    # time this section
    start_time = time.time()
    fill_storage_parallel(storage_target, files_list_targets, var_target, mean_target, std_target, "target")
    end_time = time.time()
    print(f"    Time taken to load target files: {end_time - start_time} seconds")


    storage_input = np.zeros(storage_target.shape, dtype=storage_target.dtype)

    start_time = time.time()
    fill_storage_parallel(storage_input, files_list_inputs, var_input, mean_input, std_input, "input")
    end_time = time.time()
    print(f"    Time taken to load input files: {end_time - start_time} seconds")

    print(f"    All files loaded, saving to pytorch format")
    torch_ds = torch.utils.data.TensorDataset(torch.Tensor(storage_input), torch.Tensor(storage_target))
    print(f"    Dataset shape: {torch_ds.tensors[0].shape}, {torch_ds.tensors[1].shape}")
    torch.save(torch_ds, output_file_name)
    print(f"    Dataset saved in {output_file_name}")

    # # load dataset and check that it is the same as the one we have saved
    # loaded_ds = torch.load(output_file_name, weights_only=False)
    # print(f"    Loaded dataset shape: {loaded_ds.tensors[0].shape}, {loaded_ds.tensors[1].shape}")
    # if not torch.equal(torch_ds.tensors[0], loaded_ds.tensors[0]) or not torch.equal(torch_ds.tensors[1], loaded_ds.tensors[1]):
    #     print(f"Error: the loaded dataset is different from the one we have saved")
    #     sys.exit(1)
    # print(f"    Dataset loaded successfully, check passed")


