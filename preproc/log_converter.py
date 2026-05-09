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


def 

def main():
    print("Hello, World!")
    file_path = "/leonardo_scratch/large/userexternal/gzuccari/iCMS_nc/chl/chl_2006-000.nc"
    output_path = "/leonardo/home/userexternal/gzuccari/git/OGS/SR_Unet/preproc/"
    # read the file at the input path
    ds = nc.Dataset(file_path)
    # create a perfect copuy of the file at the output path
    ds_out = nc.Dataset(os.path.join(output_path, "chl_2006-000.nc"), "w", format="NETCDF4")
    # copy the dimensions
    for name, dimension in ds.dimensions.items():
        ds_out.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
    # copy the variables
    for name, variable in ds.variables.items():
        out_var = ds_out.createVariable(name, variable.datatype, variable.dimensions)
        out_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
        data = variable[:]
        if name in ds.dimensions:
            print(f"Variable '{name}' is a dimension, copying data without log transformation.")
            out_var[:] = data
        elif np.issubdtype(np.dtype(variable.datatype), np.number):
            print(f"Variable '{name}' is numeric, applying log transformation.")
            out_var[:] = np.log(data)
        else:
            out_var[:] = data
    # close the datasets
    ds.close()
    ds_out.close()
    print("File copied successfully!")


if __name__ == "__main__":
    main()



