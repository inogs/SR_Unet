import netCDF4 as nc
import torch
import numpy as np
import os
import sys
import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from types import SimpleNamespace


def parse_input_parameters():

    parser = argparse.ArgumentParser(
        description="Produce a torch dataset from config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf_make.json"),
        help="Path to configuration file.",
    )
    args = parser.parse_args()

    return args


def read_file_list(file_path):
    files = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            files.append(line.strip())
    return files


def read_stats_file(stat_path):
    with open(stat_path, 'r') as f:
        stats = [line.strip() for line in f if line.strip()]
    mean_value = float(stats[0])
    std_value = float(stats[1])
    return mean_value, std_value


def validate_conf_file_path(conf_path):
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")
    if not os.path.isfile(conf_path):
        raise ValueError(f"Configuration path is not a file: {conf_path}")
    print(f"Configuration path check: passed")
    print(f"    Configuration path: {conf_path}")


def validate_conf(conf):
    required_path_fields = [
        "path_target",
        "path_input",
        "stat_target",
        "stat_input",
    ]
    required_fields = required_path_fields + [
        "var_target",
        "var_input",
        "output_path",
        "output_file_name",
        "n_workers",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    for field_name in required_path_fields:
        field_value = getattr(conf, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"Configuration field must be a non-empty path string: {field_name}")

    for path in [conf.path_target, conf.path_input, conf.stat_target, conf.stat_input]:
        if not os.path.exists(path):
            print(f"Error: the path {path} does not exist")
            sys.exit(1)
        else:
            print(f"    Path {path} exists, check passed")

    print("Configuration file check: passed")
    for field_name, field_value in sorted(vars(conf).items()):
        print(f"    {field_name}: {field_value}")


def read_conf_file(conf_path):
    with open(conf_path, 'r') as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def load_slice(index, file_path, variable, mean_value, std_value):
    with nc.Dataset(file_path) as dataset:
        array = dataset[variable][:]
        array_mask = np.ma.getmaskarray(array)
        valid_mask = ~array_mask
        data_slice = np.zeros(array.shape, dtype=np.result_type(array.dtype, np.float32))
        data_slice[valid_mask] = (array.data[valid_mask] - mean_value) / std_value

    return index, data_slice


def fill_storage_parallel(storage, files, variable, mean_value, std_value, label, n_workers):
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

    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    validate_conf(conf)

    output_file_path = os.path.join(conf.output_path, conf.output_file_name)
    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists, skipping pt creation.")
        sys.exit(0)

    files_list_targets = read_file_list(conf.path_target)
    files_list_inputs = read_file_list(conf.path_input)
    # check that the number of target and input files match
    if len(files_list_targets) != len(files_list_inputs):
        raise ValueError(f"Number of target files ({len(files_list_targets)}) does not match number of input files ({len(files_list_inputs)})")
    n_targets = len(files_list_targets)
    print(f"Number of samples to be written on file: {n_targets}")

    # section, load semples and define the dataset container
    dataset_1 = nc.Dataset(files_list_targets[0])
    ds_1 = dataset_1[conf.var_target]
    print(f"    Shape of the variable {conf.var_target} in the first file: {ds_1.shape}")
    storage_target = np.zeros((n_targets, 1, ds_1.shape[0], ds_1.shape[1], ds_1.shape[2]), dtype=ds_1.dtype)
    storage_input = np.zeros(storage_target.shape, dtype=storage_target.dtype)
    dataset_1.close()
    del ds_1
    print(f"    Container shape", storage_target.shape)

    # section, load statistics
    mean_target, std_target = read_stats_file(conf.stat_target)
    print(f"    Mean of target variable: {mean_target}")
    print(f"    Std of target variable: {std_target}")

    mean_input, std_input = read_stats_file(conf.stat_input)
    print(f"    Mean of input variable: {mean_input}")
    print(f"    Std of input variable: {std_input}")

    # section, fill storages
    start_time = time.time()
    fill_storage_parallel(storage_target, files_list_targets, conf.var_target, mean_target, std_target, "target", conf.n_workers)
    fill_storage_parallel(storage_input, files_list_inputs, conf.var_input, mean_input, std_input, "input", conf.n_workers)
    end_time = time.time()
    print(f"    Time taken to load target and input files: {end_time - start_time} seconds")

    print(f"    All files loaded, saving to pytorch format")
    torch_ds = torch.utils.data.TensorDataset(torch.Tensor(storage_input), torch.Tensor(storage_target))
    print(f"    Dataset shape: {torch_ds.tensors[0].shape}, {torch_ds.tensors[1].shape}")
    os.makedirs(conf.output_path, exist_ok=True)
    torch.save(torch_ds, output_file_path)
    print(f"    Dataset saved in {output_file_path}")

