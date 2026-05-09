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


# define main and print hello world

def single_conversion(file_path, output_dir):

    # read the file at the input path
    ds = nc.Dataset(file_path)
    # create a perfect copuy of the file at the output path
    # get the file name from the file path
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    ds_out = nc.Dataset(output_path, "w", format="NETCDF4")

    # copy the dimensions
    for name, dimension in ds.dimensions.items():
        ds_out.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
    # copy the variables
    for name, variable in ds.variables.items():
        out_var = ds_out.createVariable(name, variable.datatype, variable.dimensions)
        out_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
        data = variable[:]
        if name in ds.dimensions:
            print(f"    Variable '{name}' is a dimension, copying data without log transformation.")
            out_var[:] = data
        elif np.issubdtype(np.dtype(variable.datatype), np.number):
            print(f"    Variable '{name}' is numeric, applying log transformation.")
            out_var[:] = np.log(data)
        else:
            out_var[:] = data
    # close the datasets
    ds.close()
    ds_out.close()
    print("File copied successfully!")


# define a function, get the list of all files in the input directory,
# check that each file is a netcdf file, store a list with only netcdf files
# the function only returns the list of netcdf files
def get_netcdf_file_list(input_dir):
    file_list = []
    for file in os.listdir(input_dir):
        if file.endswith(".nc"):
            file_list.append(os.path.join(input_dir, file))
    return file_list


if __name__ == "__main__":
    print("Starting program")
    folder_path = "/leonardo_scratch/large/userexternal/gzuccari/chl/"
    output_path = "/leonardo_scratch/large/userexternal/gzuccari/chl.log"
    print("     folder_path: ", folder_path)
    print("     output_path: ", output_path)

    file_list = get_netcdf_file_list(folder_path)
    print(f"     Found {len(file_list)} netCDF files in the folder.")

    # print the first 5 files in the list on 5 lines
    for i in range(min(5, len(file_list))):
        print(f"     {i+1}: {file_list[i]}")

    # convert the first five files in the list using the single_conversion function
    for i in range(min(5, len(file_list))):
        print(f"     Converting file {i+1}/{len(file_list)}: {file_list[i]}")
        single_conversion(file_list[i], output_path)

    print("Ending program")



