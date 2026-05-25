import os
import sys
import netCDF4 as nc
import numpy as np

from mpi4py import MPI
from bitsea.commons.geodistances import compute_geodesic_distance


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_processes = comm.Get_size()

print(f"rank: {rank}")


def compute_distance_map(cadeau_grid, copernicus_grid, var, output_path):

    CMS2OGS_MAP = {
        "chl":"Chla",
        "dissic":"DIC",
        "nh4":"N4n",
        "no3":"N3n",
        "o2":"O2o",
        "phyc":"PhyC",
        "po4":"N1p",
        "so":"S",
        "talk":"Ac",
        "thetao":"T"
    }

    var_cad = CMS2OGS_MAP[var]

    cadeau_lon = cadeau_grid['longitude'][:]
    cadeau_lat = cadeau_grid['latitude'][:]
    cadeau_depth = cadeau_grid['depth'][:]

    copernicus_lon = copernicus_grid['longitude'][:]
    copernicus_lat = copernicus_grid['latitude'][:]
    copernicus_depth = copernicus_grid['depth'][:]

    len_lat_cad = len(cadeau_lat)
    len_lon_cad = len(cadeau_lon)

    # MPI SPLIT

    rows = np.arange(len_lat_cad)

    assigned_rows = rows[rank::n_processes]

    print(f"rank {rank} works on rows: "
          f"{assigned_rows[0]} -> {assigned_rows[-1]}")

    local_distance_map = np.full(
        (len(assigned_rows), len_lon_cad),
        np.nan,
        dtype=np.float32
    )

    # 

    cad_data = cadeau_grid[var_cad][:]
    cadeau_mask = np.ma.masked_invalid(cad_data).mask

    cop_data = copernicus_grid[var][:]
    copernicus_mask = np.ma.masked_invalid(cop_data).mask

    
    # MAIN LOOP

    for local_i, i in enumerate(assigned_rows):

        print(f"rank {rank} processing row {i}")

        for j in range(len_lon_cad):

            water_column = ~cadeau_mask[:, i, j]

            if not np.any(water_column):
                continue

            last_water_k = np.where(water_column)[0][-1]

            bottom_depth = cadeau_depth[last_water_k]

            lat0 = cadeau_lat[i]
            lon0 = cadeau_lon[j]

            # trova layer copernicus compatibile

            kk = np.where(copernicus_depth >= bottom_depth)[0]

            if len(kk) == 0:
                kcop = len(copernicus_depth) - 1
            else:
                kcop = kk[0]

            # maschera acqua copernicus

            water_mask_cop = ~copernicus_mask[kcop]

            if not np.any(water_mask_cop):
                continue

            ii, jj = np.where(water_mask_cop)

            min_dist = np.inf

            for ic, jc in zip(ii, jj):

                latc = copernicus_lat[ic]
                lonc = copernicus_lon[jc]

                dist = compute_geodesic_distance(
                    lat1=lat0,
                    lon1=lon0,
                    lat2=latc,
                    lon2=lonc
                )

                if dist < min_dist:
                    min_dist = dist

            local_distance_map[local_i, j] = min_dist


    # SAVE LOCAL RESULT

    os.makedirs(output_path, exist_ok=True)

    outfile = os.path.join(
        output_path,
        f"distance_map_{var}_rank{rank}.npy"
    )

    np.save(outfile, local_distance_map)

    # salva anche le righe associate

    rowsfile = os.path.join(
        output_path,
        f"rows_{var}_rank{rank}.npy"
    )

    np.save(rowsfile, assigned_rows)

    print(f"rank {rank} saved {outfile}")


if __name__ == "__main__":

    i = 1

    var_list = []

    input_path_cad = None
    input_path_cop = None
    output_path = None

    while i < len(sys.argv):

        if sys.argv[i] == "-ip_cad":
            input_path_cad = sys.argv[i+1]
            i += 2

        elif sys.argv[i] == "-ip_cop":
            input_path_cop = sys.argv[i+1]
            i += 2

        elif sys.argv[i] == "-op":
            output_path = sys.argv[i+1]
            i += 2

        elif sys.argv[i] == "-v":

            while i < len(sys.argv)-1:

                if not sys.argv[i+1].startswith("-"):
                    var_list.append(sys.argv[i+1])
                    i += 1
                else:
                    break

            i += 1

        else:
            i += 1

    with nc.Dataset(input_path_cad) as cad_nc, \
         nc.Dataset(input_path_cop) as cop_nc:

        for var in var_list:

            print(f"rank {rank} starts {var}")

            compute_distance_map(
                cad_nc,
                cop_nc,
                var,
                output_path
            )

            print(f"rank {rank} ends {var}")