import argparse
import os
import netCDF4 as nc
import numpy as np
import numpy.ma as ma

def parse_input_parameters():

    parser = argparse.ArgumentParser(
        description="Split the dataset into train and test files."
    )
    # parse following arguments:
    # -data_path (string): path of the data files (mandatory)
    # -variable name (string): variable to extract from the data files (mandatory)
    parser.add_argument("-dp", "--file-path", required=True, help="Path of the file list")
    parser.add_argument("-v", "--variable", required=True, help="Variable to extract from the data files")

    args = parser.parse_args()    

    return args

if __name__ == '__main__':

    print("Hello World!")
    args = parse_input_parameters()
    print(f"File path: {args.file_path}")
    print(f"Variable: {args.variable}")

    # open the file, read every line and store it in a list
    file_list = []
    with open(args.file_path, 'r') as f:
        file_list = f.readlines()
    # print first 5 elements and total number
    print(f"Total number of files: {len(file_list)}")
    # print(f"First 5 files: {file_list[:5]}")

    # 

    first_file = file_list[0].strip()
    print(f"First file: {first_file}")
    file_ds = nc.Dataset(first_file)
    print(f"Variables in the file: {file_ds.variables.keys()}")

    # get variables list
    var_list = list(file_ds.variables.keys())
    print(f"Variables list: {var_list}")

    # check if the args.variable in in the var_list, return
    # error otherwise
    if args.variable not in var_list:
        raise ValueError(f"Variable {args.variable} not found in the file. Available variables: {var_list}")
    
    array = file_ds[args.variable][:]
    print(f"Shape of the variable array: {array.shape}")

    # define a timer here
    import time

    num_nan = np.isnan(array).sum()
    print(f"Number of NAN in the variable array: {num_nan}")
    

    # define list, cycle over the file list
    # append the variable array to the list, concatenate it and compute mean and std
    l = []
    start_time = time.time()
    for filename in file_list:
        file_path = filename.strip()
        file_ds = nc.Dataset(file_path)
        l.append(file_ds[args.variable][:])
    end_time = time.time()
    print(f"Time taken to read and append variable arrays: {end_time - start_time} seconds")
    l1 = ma.concatenate(l)
 