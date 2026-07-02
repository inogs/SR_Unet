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
n_processes = comm.Get_size()


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Apply a lower treshold to a netCDF variable using MPI."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf_apply_treshold_mpi.json"),
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
        "variable_name",
        "treshold",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    for field_name in ["folder_path", "output_path", "variable_name"]:
        field_value = getattr(conf, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"Configuration field must be a non-empty string: {field_name}"
            )

    if not isinstance(conf.treshold, (int, float)):
        raise ValueError("Configuration field treshold must be numeric")

    if not os.path.exists(conf.folder_path):
        raise FileNotFoundError(f"Input folder not found: {conf.folder_path}")
    if not os.path.isdir(conf.folder_path):
        raise ValueError(f"Input path is not a folder: {conf.folder_path}")
    if os.path.exists(conf.output_path) and not os.path.isdir(conf.output_path):
        raise ValueError(f"Output path is not a folder: {conf.output_path}")

    if rank == 0:
        print("Configuration file check: passed")
        for field_name, field_value in sorted(vars(conf).items()):
            print(f"    {field_name}: {field_value}")


def apply_treshold(file_path, output_dir, variable_name, treshold):
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    file_name = f"{name}.treshold{ext}"

    output_path = os.path.join(output_dir, file_name)
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists, skipping conversion.")
        return

    ds = nc.Dataset(file_path)

    temp_output_path = os.path.join(output_dir, f"temp_{rank}_{file_name}")
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

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
    converted = ma.maximum(ma.array(out_var[:]), treshold)
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


if __name__ == "__main__":
    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    validate_conf(conf)

    folder_path = conf.folder_path
    output_path = conf.output_path
    variable_name = conf.variable_name
    treshold = conf.treshold

    if rank == 0 and not os.path.exists(output_path):
        os.makedirs(output_path)
    comm.Barrier()

    file_list = get_netcdf_file_list(folder_path)
    validate_file_list(file_list, variable_name)

    if rank == 0:
        print("This is the master process.")
        print(f"Found {len(file_list)} netCDF files in the folder.")

    assigned_data_files = file_list[rank::n_processes]

    for file in assigned_data_files:
        print(f"Process {rank} assigned file: {file}")

    for file in assigned_data_files:
        print(f"Process {rank} converting file: {file}")
        apply_treshold(
            file_path=file,
            output_dir=output_path,
            variable_name=variable_name,
            treshold=treshold,
        )
