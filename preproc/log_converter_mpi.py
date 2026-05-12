from mpi4py import MPI
import argparse
import json
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import os
from types import SimpleNamespace

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = comm.Get_size()


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Convert a netCDF variable using MPI from a config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf_convert.json"),
        help="Path to configuration file.",
    )
    return parser.parse_args()


def validate_conf_file_path(conf_path):
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")
    if not os.path.isfile(conf_path):
        raise ValueError(f"Configuration path is not a file: {conf_path}")

    if rank == 0:
        print("Configuration path check: passed")
        print(f"    Configuration path: {conf_path}")


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def validate_conf(conf):
    required_fields = [
        "folder_path",
        "output_path",
        "conversion_type",
        "variable_name",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    for field_name in required_fields:
        field_value = getattr(conf, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"Configuration field must be a non-empty string: {field_name}"
            )

    if not os.path.exists(conf.folder_path):
        raise FileNotFoundError(f"Input folder not found: {conf.folder_path}")
    if not os.path.isdir(conf.folder_path):
        raise ValueError(f"Input path is not a folder: {conf.folder_path}")
    if os.path.exists(conf.output_path) and not os.path.isdir(conf.output_path):
        raise ValueError(f"Output path is not a folder: {conf.output_path}")

    # check conversion typ, it has to be either "log" or "exp", if not raise an error
    if conf.conversion_type not in ["log", "exp"]:
        raise ValueError(
            "Configuration field conversion_type must be one of: exp, log"
        )

    if rank == 0:
        print("Configuration file check: passed")
        for field_name, field_value in sorted(vars(conf).items()):
            print(f"    {field_name}: {field_value}")


def single_conversion(file_path, conversion_type, conversion_function, output_dir, variable_name):

    # -- Section: file handling
    ds = nc.Dataset(file_path)
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)

    #instead of .log put conversion type in the file name
    file_name = f"{name}.{conversion_type}{ext}"
    
    # create new file path and check if it already exists, if it does skip the conversion
    output_path = os.path.join(output_dir, file_name)
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists, skipping conversion.")
        return

    # create a temporary output file path for the current process, if it already exists remove it
    temp_output_path = os.path.join(output_dir, f"temp_{rank}_{file_name}")
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    # -- Section: conversion
    ds_out = nc.Dataset(temp_output_path, "w", format="NETCDF4")

    for attr_name in ds.ncattrs():
        ds_out.setncattr(attr_name, ds.getncattr(attr_name))

    for dim_name, dimension in ds.dimensions.items():
        ds_out.createDimension(
            dim_name,
            len(dimension) if not dimension.isunlimited() else None,
        )

    for var_name, src_var in ds.variables.items():
        if "_FillValue" in src_var.ncattrs():
            out_var = ds_out.createVariable(
                var_name,
                src_var.datatype,
                src_var.dimensions,
                fill_value=src_var.getncattr("_FillValue"),
            )
        else:
            out_var = ds_out.createVariable(
                var_name,
                src_var.datatype,
                src_var.dimensions,
            )

        for attr_name in src_var.ncattrs():
            if attr_name != "_FillValue":
                out_var.setncattr(attr_name, src_var.getncattr(attr_name))

        out_var[:] = src_var[:]

    out_var = ds_out.variables[variable_name]
    converted = conversion_function(ma.array(out_var[:]))
    out_var[:] = converted

    if converted.count() > 0:
        data_min = float(ma.min(converted))
        data_max = float(ma.max(converted))

        if "valid_min" in out_var.ncattrs():
            out_var.setncattr("valid_min", data_min)
        if "valid_max" in out_var.ncattrs():
            out_var.setncattr("valid_max", data_max)
        if "actual_range" in out_var.ncattrs():
            out_var.setncattr(
                "actual_range",
                np.array([data_min, data_max], dtype=np.float32),
            )

    ds.close()
    ds_out.close()

    # rename the temporary output file to the final output file
    os.rename(temp_output_path, output_path)
    print("File copied successfully!")

def get_netcdf_file_list(input_dir):
    file_list = []
    for file in os.listdir(input_dir):
        if file.endswith(".nc"):
            file_list.append(os.path.join(input_dir, file))
    return sorted(file_list)


def validate_file_list(file_list, variable_name):
    if not file_list:
        raise ValueError("No netCDF files found in the input folder.")

    with nc.Dataset(file_list[0]) as dataset:
        if variable_name not in dataset.variables:
            raise ValueError(
                f"Variable {variable_name} not found in first netCDF file: {file_list[0]}"
            )


# define a function that takes a string as argument 

def get_conversion_function(conversion_type):
    if conversion_type == "log":
        return np.log
    elif conversion_type == "exp":
        return np.exp
    return None

if __name__ == "__main__":

    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    validate_conf(conf)

    folder_path = conf.folder_path
    output_path = conf.output_path
    conversion_type = conf.conversion_type
    variable_name = conf.variable_name
    conversion_function = get_conversion_function(conversion_type)

    # chek if output path exists, if not create it
    if rank == 0 and not os.path.exists(output_path):
        os.makedirs(output_path)
    comm.Barrier()
    
    file_list = get_netcdf_file_list(folder_path)
    validate_file_list(file_list, variable_name)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("This is the master process.")
        print(f"Found {len(file_list)} netCDF files in the folder.")

    # process specific instructions
    assigned_data_files = file_list[rank::n_processes]

    for file in assigned_data_files:
        print(f"Process {rank} assigned file: {file}")

    for file in assigned_data_files:
        print(f"Process {rank} converting file: {file}")
        single_conversion(file_path = file,
                        conversion_type = conversion_type,
                        conversion_function = conversion_function,
                        output_dir = output_path,
                        variable_name = variable_name)
# end of main
# end of file
