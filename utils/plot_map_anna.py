import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
import cmocean
import os

# Mappa pred/copernics → target (cadeau/ogs)
CMS2OGS_MAP = {
    "chl": "Chla",
    "dissic": "DIC",
    "nh4": "N4n",
    "no3": "N3n",
    "o2": "O2o",
    "phyc": "PhyC",
    "po4": "N1p",
    "so": "S",
    "talk": "Ac",
    "thetao": "T"
}

def plot_netcdf(file_target, file_pred, var_pred_name, abs_diff=False):

    if var_pred_name not in CMS2OGS_MAP:
        raise ValueError(f"Variabile '{var_pred_name}' non trovata nel mapping.")

    var_target_name = CMS2OGS_MAP[var_pred_name]

    print(f"Pred variable: {var_pred_name}")
    print(f"Target variable: {var_target_name}")

    # --- TARGET ---
    ds_target = nc.Dataset(file_target, 'r')
    var_target = ds_target.variables[var_target_name][:]

    lon = ds_target.variables['longitude'][:]
    lat = ds_target.variables['latitude'][:]

    # --- PRED ---
    ds_pred = nc.Dataset(file_pred, 'r')
    var_pred = ds_pred.variables[var_pred_name][:]

    # Numero righe
    nrows = 2 + (abs_diff)
    fig, axes = plt.subplots(nrows=nrows, figsize=(16, 6*nrows), constrained_layout=True)

    # Range comune per confronto corretto
    vmax = max(np.nanmax(var_target), np.nanmax(var_pred)) # per il range dei colori prendo il massimo tra il range pred e ogs e uso quello oer entrambi i plot così che sono comparabili
    vmin = min(np.nanmin(var_target), np.nanmin(var_pred)) # in verità non serve eprchè metto zero MA leggi commento cmap.set_under('red')

    cmap = cmocean.cm.dense.copy()
    cmap.set_under('red') # colora i pixel di rosso se ci sono valori sotto quello che metto come minimo in .pcolormesh (cioè zero)

    # --- 1) TARGET ---
    im0 = axes[0].pcolormesh(
        lon, lat, var_target[0,:,:], # var_target[0,:,:] prende solo la superficie (depth=0)
        cmap=cmap, vmin=0, vmax=vmax
    )
    file_name = os.path.basename(file_target)  
    date_str = file_name[-11:] 
    axes[0].set_title(f"Cadeau - {date_str}", fontsize=18)
    fig.colorbar(im0, ax=axes[0])
    axes[0].tick_params 

    # --- 2) PRED ---
    im1 = axes[1].pcolormesh(
        lon, lat, var_pred[0,:,:],
        cmap=cmap, vmin=0, vmax=vmax
    )
    axes[1].set_title("Prediction", fontsize=18)
    fig.colorbar(im1, ax=axes[1])
    axes[0].tick_params

    # --- 3) DIFFERENZA ASSOLUTA ---
    if abs_diff:
        diff = np.abs(var_target[0,:,:] - var_pred[0,:,:]) # mantiene la corrispondenza pixel per pixel perché NumPy fa operazioni elemento per elemento (“broadcasting”) su array dello stesso shape. 
        vmax_diff = np.nanmax(diff)
       
        im2 = axes[2].pcolormesh(
            lon, lat, diff,
            cmap='magma',
            vmin=0,
            vmax=vmax_diff
        )
        axes[2].set_title("Absolute Difference", fontsize=18)
        fig.colorbar(im2, ax=axes[2])
        axes[0].tick_params 
        
        # calcolo % abs diff > 1 e % abs diff > 2 
        perc_magg_1 = np.sum((diff > 1) & (~np.isnan(diff)))/diff.size*100
        perc_magg_2 = np.sum((diff > 2) & (~np.isnan(diff)))/diff.size*100
        print(f"% values |diff| > 1: {perc_magg_1:.2f}%, > 2: {perc_magg_2:.2f}%")
        axes[2].text(
            x=0.70, y=0.95,  # coordinate in frazione dell'asse (0-1)
            s=f"Absolute Difference\n>1: {perc_magg_1:.2f}%, >2: {perc_magg_2:.2f}%",
            fontsize=18,
            color='black',
            transform=axes[2].transAxes,  # importante: coord in unità dell'asse
            verticalalignment='top'
         )


    ds_target.close()
    ds_pred.close()

    plt.savefig("comparison_plot.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage:")
        print("python script.py <pred_var> <target_file> <pred_file> [-abs]")
        sys.exit(1)

    pred_var = sys.argv[1]
    target_file = sys.argv[2]
    pred_file = sys.argv[3]

    abs_flag = "-abs" in sys.argv

    plot_netcdf(target_file, pred_file, pred_var, abs_flag)