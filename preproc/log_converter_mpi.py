from mpi4py import MPI
import numpy as np
import netCDF4 as nc
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = comm.Get_size()

def single_conversion(file_path, output_dir):

    ds = nc.Dataset(file_path)
    file_name = os.path.basename(file_path)
    # add log. to the file name before the extension
    name, ext = os.path.splitext(file_name)
    file_name = f"{name}.log{ext}"
    output_path = os.path.join(output_dir, file_name)

    # check if the output file already exists, if it does delete exit the function
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists, skipping conversion.")
        return
    # if os.path.exists(output_path):
    #     print(f"Output file {output_path} already exists, deleting it.")
    #     os.remove(output_path)

    ds_out = nc.Dataset(output_path, "w", format="NETCDF4")

    for name, dimension in ds.dimensions.items():
        ds_out.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    for name, variable in ds.variables.items():
        out_var = ds_out.createVariable(name, variable.datatype, variable.dimensions)
        out_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
        data = variable[:]
        if name in ds.dimensions:
            # print(f"    Variable '{name}' is a dimension, copying data without log transformation.")
            out_var[:] = data
        elif np.issubdtype(np.dtype(variable.datatype), np.number):
            # print(f"    Variable '{name}' is numeric, applying log transformation.")
            out_var[:] = np.log(data)
        else:
            out_var[:] = data
    # close the datasets
    ds.close()
    ds_out.close()
    print("File copied successfully!")

def print_hello():
    print(f"Hello world from rank {rank} of {size}")

def get_netcdf_file_list(input_dir):
    file_list = []
    for file in os.listdir(input_dir):
        if file.endswith(".nc"):
            file_list.append(os.path.join(input_dir, file))
    return file_list


if __name__ == "__main__":
    print_hello()

    # folder_path = "/leonardo_work/OGS23_PRACE_IT_0/fadobbat/working_data/NARF_nc/Chla/"
    folder_path = "/leonardo_scratch/large/userexternal/gzuccari/NARF_nc/Chla/"
    output_path = "/leonardo_scratch/large/userexternal/gzuccari/NARF_nc/Chla.log/"
    file_list = get_netcdf_file_list(folder_path)

    # make an if on rank 0
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("This is the master process.")
        # get the list of netcdf files in the folder
        print(f"Found {len(file_list)} netCDF files in the folder.")

    assigned_data_files = file_list[rank::n_processes]

    for file in assigned_data_files:
        print(f"Process {rank} assigned file: {file}")

    # convert the assigned files
    for file in assigned_data_files:
        print(f"Process {rank} converting file: {file}")
        single_conversion(file, output_path)