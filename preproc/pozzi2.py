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

def compute_distance_map(cadeau_grid, copernicus_grid, var, output_path):

    CMS2OGS_MAP = {"chl":"Chla", "dissic":"DIC", "nh4":"N4n", "no3":"N3n", "o2":"O2o", "phyc":"PhyC", "po4":"N1p", "so":"S", "talk":"Ac", "thetao":"T"}
    var_cad = CMS2OGS_MAP[var]

    cadeau_lon = cadeau_grid['longitude'][:]
    cadeau_lat = cadeau_grid['latitude'][:]
    cadeau_depth = cadeau_grid['depth'][:]

    copernicus_lon = copernicus_grid['longitude'][:]
    copernicus_lat = copernicus_grid['latitude'][:]
    copernicus_depth = copernicus_grid['depth'][:]

    len_lat_cad =  len(cadeau_lat)
    len_lon_cad = len(cadeau_lon)
    distance_map = np.full((len_lat_cad, len_lon_cad), np.nan)


    # dati e maschera cadeau
    cad_data = cadeau_grid[var_cad][:]
    cadeau_mask = np.ma.masked_invalid(cad_data).mask

    # dati e maschera copernicus
    cop_data = copernicus_grid[var][:]
    copernicus_mask = np.ma.masked_invalid(cop_data).mask

    for i in range(len_lat_cad):
        for j in range(len_lon_cad):            
            # salto punti superficiali
            # if cadeau_mask[0, i, j]:
            #     continue               

            # 1. TROVO FONDO CADEAU
            # trovo l'ultima cella di acqua in cadeau
            water_column = ~cadeau_mask[:, i, j] # true = acqua

            if not np.any(water_column): # esiste almeno una cella d’acqua?
                continue # se False, cioè tutta terra -> salto quel punto (i,j) perché non contiene acqua

            last_water_k = np.where(water_column)[0][-1] # ultima posizione della colonnadove ho acqua
            # print('last_water_k', last_water_k)

            bottom_depth = cadeau_depth[last_water_k] # prendo i metri corrispondenti
            # print('bottom_depth', bottom_depth)

            lat0 = cadeau_lat[i]
            lon0 = cadeau_lon[j]

            # 2. TROVO LAYER COPERNICUS PIÙ PROFONDO DEL FONDO
            kk = np.where(copernicus_depth >= bottom_depth)[0]
            # print('layer copernicus più profondo del bottom di cadeau', kk)

            if len(kk) == 0: # caso limite: il fondo Cadeau è più profondo dell’ultimo livello Copernicus
                kcop = len(copernicus_depth) - 1 # prendo l'ultimo layer dsiponibile cioè il più profondo che ho
            else:
                kcop = kk[0] # altrimenti prendo il PRIMO layer sufficientemente profondo

            
            # 3. PUNTI ACQUA COPERNICUS
            # tutti i punti orizzontali di Copernicus che sono acqua al layer kcop
            water_mask_cop = ~copernicus_mask[kcop]

            if not np.any(water_mask_cop):
                continue

            ii, jj = np.where(water_mask_cop) # le coordinate di tutti i punti acqua

            # 4. DISTANZA MINIMA
            min_dist = np.inf

            for ic, jc in zip(ii, jj): # per ogni punto di acqua copernicus (ii,jj) a quella profondità kcop

                # uso lat lon reali per calcolare le distanze così lo faccio tra i CENTRI delle celle
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

            distance_map[i, j] = min_dist

    # SAVE DISTANCE MAP
    os.makedirs(output_path, exist_ok=True)
    outfile = os.path.join(
        output_path,
        f"distance_map_{var}.txt"
        )

    np.savetxt(
        outfile,
        distance_map,
        fmt="%.3f"
    )

    print(f"Distance map saved in: {outfile}")
            
    return distance_map




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
            dist = compute_distance_map(cad_nc, cop_nc, var, output_path)
            print(dist)
            print(f"Ending execution")
