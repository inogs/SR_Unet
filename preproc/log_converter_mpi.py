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
    name, ext = os.path.splitext(file_name)
    file_name = f"{name}.log{ext}"
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists, skipping conversion.")
        return

    temp_output_path = os.path.join(output_dir, f"temp_{rank}_{file_name}")
    ds_out = nc.Dataset(temp_output_path, "w", format="NETCDF4")

    for name, dimension in ds.dimensions.items():
        ds_out.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    for name, variable in ds.variables.items():
        out_var = ds_out.createVariable(name, variable.datatype, variable.dimensions)
        out_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
        data = variable[:]
        if name in ds.dimensions:
            out_var[:] = data
        elif np.issubdtype(np.dtype(variable.datatype), np.number):
            out_var[:] = np.log(data)
        else:
            out_var[:] = data

    ds.close()
    ds_out.close()

    os.rename(temp_output_path, output_path)
    print("File copied successfully!")

def get_netcdf_file_list(input_dir):
    file_list = []
    for file in os.listdir(input_dir):
        if file.endswith(".nc"):
            file_list.append(os.path.join(input_dir, file))
    return file_list


if __name__ == "__main__":

    folder_path = "/leonardo_scratch/large/userexternal/gzuccari/NARF_nc/Chla/"
    output_path = "/leonardo_scratch/large/userexternal/gzuccari/NARF_nc/Chla.log.v2/"
    # chek if output path exists, if not create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    file_list = get_netcdf_file_list(folder_path)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("This is the master process.")
        # get the list of netcdf files in the folder
        print(f"Found {len(file_list)} netCDF files in the folder.")

    assigned_data_files = file_list[rank::n_processes]

    for file in assigned_data_files:
        print(f"Process {rank} assigned file: {file}")

    for file in assigned_data_files:
        print(f"Process {rank} converting file: {file}")
        single_conversion(file, output_path)