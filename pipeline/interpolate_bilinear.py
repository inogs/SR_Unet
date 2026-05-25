import os
import argparse
import json
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from typing import Dict
from types import SimpleNamespace
from scipy.interpolate import interpn, make_interp_spline, griddata
import scipy.interpolate as intrp


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_processes = comm.Get_size()


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Interpolate Copernicus Marine netCDF variables on a target grid using MPI."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf_interpolate.json"),
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
        "input_path",
        "output_path",
        "grid_file_path",
        "variables",
    ]
    path_fields = [
        "input_path",
        "output_path",
        "grid_file_path",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    for field_name in path_fields:
        field_value = getattr(conf, field_name)
        if not os.path.exists(field_value):
            raise FileNotFoundError(
                f"Configuration path does not exist for {field_name}: {field_value}"
            )

    if not isinstance(conf.variables, list) or len(conf.variables) == 0:
        raise ValueError(
            "Configuration field variables must contain at least one variable"
        )

    if rank == 0:
        print("Configuration file check: passed")
        for field_name, field_value in sorted(vars(conf).items()):
            print(f"    {field_name}: {field_value}")


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


def interpolate_3d(values2interp, old_lon, old_lat, old_dep, new_grid, var_grid):
    
    new_lon = new_grid['longitude'][:]
    new_lat = new_grid['latitude'][:]
    new_dep = new_grid['depth'][:]

    # dati e maschera nuov (cadeau)
    new_data = new_grid[var_grid][:]
    new_mask = np.ma.masked_invalid(new_data).mask

    # maschera dati vecchi (copernicus)
    masked_data = np.ma.masked_invalid(values2interp)   # Mask an array where invalid values occur (NaNs or infs).
    # Create a mask of valid (non-masked) points
    valid_mask = ~masked_data.mask
    
    assert values2interp.shape == (len(old_dep), len(old_lat), len(old_lon))

    # dimensioni griglia vecchia (copernicus)
    n_dep_old = (len(old_dep))
    n_lat_new = len(new_lat)
    n_lon_new = len(new_lon)

    # griglie nuova (cadeau) e vecchia (copernicus)
    LAT_new, LON_new = np.meshgrid(new_lat, new_lon, indexing="ij")
    LAT_old, LON_old = np.meshgrid(old_lat, old_lon, indexing="ij")

    # punti (lat, lon) messi come lista, griglia nuova
    points_new = np.column_stack([LAT_new.ravel(), LON_new.ravel()])

    # STEP 1: INTERPOLAZIONE ORIZZONTALE
    tmp_lin = np.empty((n_dep_old, n_lat_new, n_lon_new))
    tmp_nn = np.empty_like(tmp_lin)

    # itero su tutte le profonidtà della griglia vecchia
    for k in range(n_dep_old):
        # STEP 1.1: LINEAR -> non serve che io distingua dati validi da dati non validi -> se non ho un punto circondato completamente da acqua avrò nan
        tmp_lin[k] = interpn(
            (old_lat, old_lon),
            values2interp[k],
            points_new,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        ).reshape(n_lat_new, n_lon_new)

        # STEP 1.2: NEAREST -> qui devo considerare solo i dati validi così da avere sempre un valore != nan, tutti i punti della griglia vengono riempiti dal valore di acqua valido più vicino
        slice_data = masked_data[k]

        mask = valid_mask[k]
        values_valid = slice_data[mask]

        points_valid = np.column_stack([
            LAT_old[mask],
            LON_old[mask]
        ])

        tmp_nn[k] = griddata(
            points_valid,
            values_valid,
            points_new,
            method="nearest"
        ).reshape(n_lat_new, n_lon_new)


    # STEP 2: MERGE
    tmp_lin[tmp_lin > 1e20] = np.nan

    # metto il valore di linear se esiste altrimenti metto quello di nearest
    tmp = np.where(np.isnan(tmp_lin), tmp_nn, tmp_lin)

    # STEP 3: INTERPOLAZIONE VERTICALE -> lineare
    out = np.empty((len(new_dep), n_lat_new, n_lon_new))

    for i in range(n_lat_new):
        for j in range(n_lon_new):

            profile = tmp[:, i, j]

            # CORNER CASE -> gestione dei pozzi
            # dove la griglia fine è più profonda della grossolana, propago verso il basso l’ultimo valore oceanico valido disponibile
            profile = profile.copy()
            valid = np.isfinite(profile)

            # fill sotto
            last_valid = np.where(valid)[0][-1]
            profile[last_valid+1:] = profile[last_valid]

            # spline
            spline = make_interp_spline(
                old_dep,
                profile,
                k=1
            )

            y = spline(new_dep)

            # CORNER CASE -> gestione dei valori negativi in superficie e fvìalori fuori range in profondità
            # extrapolazione costante ai bordi:
            # sopra la superficie uso il primo valore disponibile,
            # sotto il fondo uso l'ultimo valore disponibile
            y[new_dep < old_dep.min()] = profile[0]
            y[new_dep > old_dep.max()] = profile[-1]

            out[:, i, j] = y

    # STEP 4: MASK FINALE -> applico la maschera di cadeau
    interp_data = ma.masked_array(
        out,
        mask=new_mask,
        fill_value=1e20,
        dtype=np.float32
    )

    

    return interp_data



def interpolate_data(cms_name: str, input_path: str, output_path: str, grid_file: str, var_grid:str, n_dim: int = 3) -> None:
    '''
        Performs the interpolation of a given variable and saves the results into a different folder.

        Args:
            cms_name (str): variable name in the Copernicus Marine format.
            input_path (str): directory containing Copernicus Marine data.
            output_path (str): directory where the interpolated files are saved
            grid_file (str): path to thethe NetCDF file that contains the grid for interpolation.
            var_grid (str): variable in the grid file of which we want to copy the shape
            n_dim (int): number of dimensions of the data; def 3 (dep, lat, lon)
    '''


    os.makedirs(output_path, exist_ok=True)

    # Open the grid file to get the new dimensions
    with nc.Dataset(grid_file, "r") as grid_nc:
        new_longitudes = grid_nc['longitude'][:]
        new_latitudes = grid_nc['latitude'][:]  # New latitude and longitude values
        new_depth = grid_nc['depth'][:]

        data_files = tuple(get_netcdf_file_list(input_path))
        validate_file_list(data_files, cms_name)

        assigned_data_files = data_files[rank::n_processes]

        # Iterate over the NetCDF files in the source directory
        for file_path in assigned_data_files:
            # Open the original NetCDF file for reading
            with nc.Dataset(file_path, "r") as source_nc:
                source_file_name = os.path.basename(file_path)
                old_lon = source_nc['longitude'][:]
                old_lat = source_nc['latitude'][:]
                old_dep = source_nc['depth'][:]
                if cms_name not in source_nc.variables:
                    raise ValueError(f"Variable {cms_name} not found in {file_path}")
                old_data = source_nc[cms_name][:]
                # Define the path for the modified version in the destination directory
                # file finale
                final_file_path = os.path.join(output_path, source_file_name)
                # file temporaneo
                tmp_file_path = final_file_path + ".tmp"
                
                if not os.path.exists(final_file_path):
                    # crea la cartella senza race condition MPI
                    os.makedirs(output_path, exist_ok=True)
                    # rimuovi eventuale tmp incompleto
                    if os.path.exists(tmp_file_path):
                        print(f"Removing incomplete file: {tmp_file_path}")
                        os.remove(tmp_file_path)

                # Create a new NetCDF file for writing
                    with nc.Dataset(tmp_file_path, "w") as dest_nc:
                        # Create dimensions in the new file based on the interpolated grid
                        dest_nc.createDimension("depth", len(new_depth))
                        dest_nc.createDimension("longitude", len(new_longitudes))
                        dest_nc.createDimension("latitude", len(new_latitudes))

                        # Create depth, latitude, and longitude variables in the new file
                        dest_depth = dest_nc.createVariable("depth", new_depth.dtype, ("depth",))
                        dest_longitudes = dest_nc.createVariable("longitude", new_longitudes.dtype, ("longitude",))
                        dest_latitudes = dest_nc.createVariable("latitude", new_latitudes.dtype, ("latitude",))

                        # Write the new depth, latitude, and longitude values
                        dest_depth[:] = new_depth
                        dest_longitudes[:] = new_longitudes
                        dest_latitudes[:] = new_latitudes

                        # Perform interpolation
                        interpolated_array = interpolate_3d(old_data, old_lon, old_lat, old_dep, grid_nc, var_grid)
                        # Create a variable in the new file and write the interpolated array
                        spatial_dim = ( "depth", "latitude", "longitude")
                        dest_array = dest_nc.createVariable(cms_name, interpolated_array.dtype, spatial_dim)
                        dest_array[:] = interpolated_array

                        # Copy global attributes from the original file
                        dest_nc.setncatts(source_nc.__dict__)

                        # passiamo al file definitivo
                    os.replace(tmp_file_path, final_file_path)
                    print(f"Created {final_file_path}")
                else:
                    print(f"{final_file_path} have already been created")
                






if __name__ == "__main__":
    '''
        Interpolates the files present in the source directory (with Copernicus Marine files)
        to match the dimensions with those of the files in the target directory (with CADEAU files).
        The input path must point directly to the directory containing the NetCDF files.

        Parameters:
        -c/--config path to a JSON configuration file
    '''
    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    validate_conf(conf)

    if rank == 0:
        os.makedirs(conf.output_path, exist_ok=True)
    comm.Barrier()

    for variable in conf.variables:
        print(f"[interpolate_cms for variable '{variable.name}'] Starting execution")
        interpolate_data(
            variable.name,
            conf.input_path,
            conf.output_path,
            conf.grid_file_path,
            variable.grid_variable,
            n_dim=3,
        )
        print(f"[interpolate_cms for variable '{variable.name}'] Ending execution")
