import argparse
import os
import time
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import xarray as xr

def parse_input_parameters():

    parser = argparse.ArgumentParser(
        description="Split the dataset into train and test files."
    )
    parser.add_argument("-dp", "--file-path", required=True, help="Path of the file list")
    parser.add_argument("-v", "--variable", required=True, help="Variable to extract from the data files")
    parser.add_argument("-n", type=int, default=None, help="Maximum number of files to process")

    args = parser.parse_args()    

    if args.n is not None and args.n <= 0:
        raise ValueError("-n must be a positive integer")

    return args

if __name__ == '__main__':

    args = parse_input_parameters()
    print(f"File path: {args.file_path}")
    print(f"Variable: {args.variable}")
    if args.n is not None:
        print(f"Max files to process: {args.n}")


    # section - read file list
    start_time = time.time()
    files = []
    with open(args.file_path, 'r') as f:
        for i, line in enumerate(f):
            if args.n is not None and i >= args.n:
                break
            files.append(line.strip())
    mus = []
    vars_ = []
    end_time = time.time()
    print(f"Time taken to read file list: {end_time - start_time} seconds for {len(files)} files")

    # section - compute file-wise mean and var with xarray
    start_time = time.time()
    for counter, dataset_path in enumerate(files, start=1):
        print(f"Processing file {counter}/{len(files)}: {dataset_path}")
        with xr.open_dataset(dataset_path) as dataset:
            x = dataset[args.variable]
            mus.append(x.mean().item())
            vars_.append(x.var().item())
    end_time = time.time()
    print(f"Time taken to compute file-wise mean and var with xarray: {end_time - start_time} seconds")


    # section - compute average and std with xarray
    start_time = time.time()
    mu_tot = np.mean(mus)
    sigma = np.sqrt(np.mean([
        v + (m - mu_tot)**2
        for v, m in zip(vars_, mus)
    ]))
    end_time = time.time()
    print(f"Time taken to compute average and std with xarray: {end_time - start_time} seconds")
    print("avg with xarray", mu_tot)
    print("std with xarray", sigma)

    # section - print to file
    base_name = os.path.basename(args.file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    stat_file_name = f"stat.{name_without_ext}.txt"
    stat_file_path = os.path.join(os.path.dirname(args.file_path), stat_file_name)
    with open(stat_file_path, 'w') as f:
        f.write(f"{mu_tot}\n{sigma}\n")
    print(f"Saved average and std to {stat_file_path}")
