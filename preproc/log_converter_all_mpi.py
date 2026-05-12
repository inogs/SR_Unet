#!/usr/bin/env python3
from mpi4py import MPI
import json
import os

from log_converter_mpi import (
    get_conversion_function,
    get_netcdf_file_list,
    single_conversion,
    validate_file_list,
)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
JOBS_FILE = os.path.join(os.path.dirname(__file__), "conf_convert_all.json")


with open(JOBS_FILE) as f:
    jobs = json.load(f)["jobs"]


for job_index, job in enumerate(jobs):
    folder_path = job["folder_path"]
    output_path = job["output_path"]
    conversion_type = job["conversion_type"]
    variable_name = job["variable_name"]
    conversion_function = get_conversion_function(conversion_type)

    if conversion_function is None:
        raise ValueError("conversion_type must be one of: exp, log")

    if rank == 0 and not os.path.exists(output_path):
        os.makedirs(output_path)
    comm.Barrier()

    file_list = get_netcdf_file_list(folder_path)
    validate_file_list(file_list, variable_name)

    if rank == 0:
        print(f"Running job {job_index + 1}/{len(jobs)}")
        print(f"    folder_path: {folder_path}")
        print(f"    output_path: {output_path}")
        print(f"    conversion_type: {conversion_type}")
        print(f"    variable_name: {variable_name}")
        print(f"    files: {len(file_list)}")

    assigned_data_files = file_list[rank::size]

    for file in assigned_data_files:
        print(f"Process {rank} converting file: {file}")
        single_conversion(
            file_path=file,
            conversion_type=conversion_type,
            conversion_function=conversion_function,
            output_dir=output_path,
            variable_name=variable_name,
        )

    comm.Barrier()
