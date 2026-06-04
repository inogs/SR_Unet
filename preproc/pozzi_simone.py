# quanto ogni punto Cadeau è “coperto” orizzontalmente da acqua Copernicus alla profondità del fondo

import os
import glob
import sys
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from typing import Dict
from scipy.interpolate import interpn, make_interp_spline, griddata
import scipy.interpolate as intrp
import matplotlib.pyplot as plt
import cmocean
from matplotlib.colors import ListedColormap
from bitsea.commons.geodistances import compute_geodesic_distance

def compute_coverage_map(cadeau_grid, copernicus_grid, var, output_path):

    CMS2OGS_MAP = {
        "chl":"Chla", "dissic":"DIC", "nh4":"N4n", "no3":"N3n",
        "o2":"O2o", "phyc":"PhyC", "po4":"N1p", "so":"S",
        "talk":"Ac", "thetao":"T"
    }

    var_cad = CMS2OGS_MAP[var]

    cadeau_lon = cadeau_grid['longitude'][:]
    cadeau_lat = cadeau_grid['latitude'][:]
    cadeau_depth = cadeau_grid['depth'][:]

    cop_lon = np.asarray(copernicus_grid['longitude'][:])
    cop_lat = np.asarray(copernicus_grid['latitude'][:])
    cop_depth = copernicus_grid['depth'][:]

    # mask
    cad_mask = np.ma.masked_invalid(cadeau_grid[var_cad][:]).mask
    cop_mask = np.ma.masked_invalid(copernicus_grid[var][:]).mask

    # len
    nlat_cad = len(cadeau_lat)
    nlon_cad = len(cadeau_lon)
    nlat_cop = len(cop_lat)
    nlon_cop = len(cop_lon)

    # output maps
    cad_bottom_map = np.full((nlat_cad, nlon_cad), np.nan)
    cop_bottom_map = np.full((nlat_cad, nlon_cad), np.nan)

    # profondità massima disponibile per ogni cella Copernicus
    cop_max_depth = np.full((nlat_cop, nlon_cop), np.nan)

    for i in range(nlat_cop):
        for j in range(nlon_cop):

            water_col = ~cop_mask[:, i, j]

            if np.any(water_col):

                lastk = np.where(water_col)[0][-1]

                cop_max_depth[i, j] = cop_depth[lastk]

    # loop su Cadeau

    for i in range(nlat_cad):
        for j in range(nlon_cad):

            # fondo Cadeau
            water_col = ~cad_mask[:, i, j]

            if not np.any(water_col):
                continue

            lastk = np.where(water_col)[0][-1] # prende ultimo valore di acqua


            cad_bottom_map[i, j] = cadeau_depth[lastk] # prende la profnodità in metri

            lat0 = cadeau_lat[i]
            lon0 = cadeau_lon[j]

            # le 2 latitudini Copernicus più vicine a lat0
            # le 2 longitudini Copernicus più vicine a lon0

            ilat = np.argpartition(np.abs(cop_lat - lat0), 2)[:2]
            ilon = np.argpartition(np.abs(cop_lon - lon0), 2)[:2]

        
            # prendo le 4 celle vicine

            neigh_depths = []

            for ii in ilat:
                for jj in ilon:

                    d = cop_max_depth[ii, jj]

                    if not np.isnan(d):
                        neigh_depths.append(d)

            if len(neigh_depths) == 0:
                continue

            neigh_depths = np.array(neigh_depths)

            # profondità rappresentativa Copernicus
            cop_bottom_map[i, j] = np.max(neigh_depths)

    # save

    # os.makedirs(output_path, exist_ok=True)

    # outfile_cad = os.path.join(
    #     output_path,
    #     f"cad_bottom_map_{var}.txt"
    # )

    # outfile_cop = os.path.join(
    #     output_path,
    #     f"cop_bottom_map_{var}.txt"
    # )

    # np.savetxt(outfile_cad, cad_bottom_map, fmt="%.3f")
    # np.savetxt(outfile_cop, cop_bottom_map, fmt="%.3f")

    # print(f"Cadeau bottom map saved in: {outfile_cad}")
    # print(f"Copernicus bottom map saved in: {outfile_cop}")

    return cad_bottom_map, cop_bottom_map


if __name__ == "__main__":
    '''
        Parameters:
        -ip_cad string with the path to a file with the target grid of interpolation
        -ip_cop string with the path to a file with the target grid of interpolation
        -op string with the output path (excluded variable specific dir)
        -v set of variables to interpolate
    '''
    i = 1
    var_list = []
    input_path_cad = None
    output_path = None
    input_path_cop = None

    while i < len(sys.argv):
        if sys.argv[i] == "-ip_cad":
            if input_path_cad is not None: raise ValueError("Repeated input for input path")
            input_path_cad = sys.argv[i+1]
            i += 2
        if sys.argv[i] == "-ip_cop":
            if input_path_cop is not None: raise ValueError("Repeated input for input path")
            input_path_cop = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-op":
            if output_path is not None: raise ValueError("Repeated input for output path")
            output_path = sys.argv[i+1]
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

    if input_path_cop is None: raise TypeError("Missing value for input path copernicus")
    if input_path_cad is None: raise TypeError("Missing value for input path cadeau")

    if output_path is None: raise TypeError("Missing value for output path")

    if var_list == []: raise TypeError("Missing value for variable")

    # Path to the file containing the grid to be used for interpolation

    with nc.Dataset(input_path_cad) as cad_nc, nc.Dataset(input_path_cop) as cop_nc:
        
        for var in var_list:
            print(f"Starting execution for {var}")
            compute_coverage_map(cad_nc, cop_nc, var, output_path)
            print(f"Ending execution")