import os
import glob
import sys
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from typing import Dict
from scipy.interpolate import interpn, make_interp_spline, griddata
import scipy.interpolate as intrp


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_processes = comm.Get_size()
print('rank:', rank)



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
    n_dep_old = (len(old_dep) -2 ) # per togliere gli ultimi due layer perchè non avevo valori
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

        # STEP 1.2: NEAREST -> qui devo considerare solo i dati validi così da avere sempre un valore != nan, tutti i punti della griglia venogno riempiti dal valore di acquà valido più vicino
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

    # Quali layer verticali della nearest interpolation (tmp_nn) contengono ancora NaN?
    problem_depths = np.where(np.isnan(tmp_nn).reshape(n_dep_old, -1).any(axis=1))[0]
    print("\n DEPTH CON NaN IN tmp_nn:", problem_depths)

    # metto il valore di linear se esiste altrimenti metto quello di nearest
    tmp = np.where(np.isnan(tmp_lin), tmp_nn, tmp_lin)

    # printo delle informazioni
    print('sum nan in tmp lin',np.sum(np.isnan(tmp_lin))) # normale che ci siano
    print('sum nan in tmp nn',np.sum(np.isnan(tmp_nn))) # deve essere 0
    print(f"Max in tmp nn: {np.nanmax(tmp_nn)}")  # deve essere un valore ragionevole
    print(f"Max in tmp lin: {np.nanmax(tmp_lin)}") # deve essere un valore ragionevole


    # STEP 3: INTERPOLAZIONE VERTICALE -> lineare
    out = np.empty((len(new_dep), n_lat_new, n_lon_new))
    old_dep = old_dep[:-2] # tolgo gli ultimi due layers


    for i in range(n_lat_new):
        for j in range(n_lon_new):

            profile = tmp[:, i, j]
            spline = make_interp_spline(
                old_dep,
                profile,
                k=1
            )

            y = spline(new_dep)

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

        data_files = tuple(glob.glob(os.path.join(input_path, "*.nc")))

        assigned_data_files = data_files[rank::n_processes]

        # Iterate over the NetCDF files in the source directory
        for file_path in assigned_data_files:
            # Open the original NetCDF file for reading
            with nc.Dataset(file_path, "r") as source_nc:
                source_file_name = os.path.basename(file_path)
                # source_ds = nc.Dataset(file_path)
                old_lon = source_nc['longitude'][:]
                old_lat = source_nc['latitude'][:]
                old_dep = source_nc['depth'][:]
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
        All the files referring to given variable are assumed to be in a directory named as the variable.

        Parameters:
        -ip string with the input path (excluded vairable specific dir)
        -op string with the output path (excluded variable specific dir)
        -gp string with the path to a file with the target grid of interpolation
        -vgp string with the var in the gp file
        -v set of variables to interpolate
    '''
    i = 1
    var_list = []
    input_path = None
    output_path = None
    grid_file_path = None
    var_grid_path = None

    while i < len(sys.argv):
        if sys.argv[i] == "-ip":
            if input_path is not None: raise ValueError("Repeated input for input path")
            input_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-op":
            if output_path is not None: raise ValueError("Repeated input for output path")
            output_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-gp":
            if grid_file_path is not None: raise ValueError("Repeated input for output path")
            grid_file_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-vgp":
            if var_grid_path is not None: raise ValueError("Repeated input for output path")
            var_grid_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-v":
            if var_list != []: raise ValueError("Repeated input for variable")
            while i < len(sys.argv) - 1:
                if not sys.argv[i+1].startswith('-'):
                    var_list.append(sys.argv[i+1])
                    i += 1
                else:
                    break
            i += 1
        else:
            i += 1

    if input_path is None: raise TypeError("Missing value for input path")
    if output_path is None: raise TypeError("Missing value for output path")
    if grid_file_path is None: raise TypeError("Missing value for grid file path")

    if var_list == []: raise TypeError("Missing value for variable")
    if var_grid_path == None: raise TypeError("Missing value for grid variable")

    # Path to the file containing the grid to be used for interpolation

    for var in var_list:
        print(f"[interpolate_cms for variable '{var}'] Starting execution")
        interpolate_data(var, os.path.join(input_path, var), os.path.join(output_path, var), grid_file_path, var_grid_path, n_dim=3)
        print(f"[interpolate_cms for variable '{var}'] Ending execution")
