# questo script serve a prendere i tre dataset val test train e per ogni file di una specifica variabile vengono contate le celle con valore negativo e viene fatto un plot che tiene traccia di queste celle all'interno della mappa dei dati in modo da avere per ogni cella mare un conteggio dei valori negativi nei vari file

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
import matplotlib.cm as cmo
import seaborn as sns
import cmocean
from matplotlib.colors import ListedColormap
import os
from scipy.ndimage import uniform_filter
import natsort
import subprocess
import matplotlib.colors as mcolors


def analyze_multi_dataset_neg(data_dirs, var_name, output_txt):

    neg_counter = None
    mask = None

    for data_dir in data_dirs:
        var_dir = os.path.join(data_dir, var_name)

        print("DIR:", var_dir)

        if not os.path.exists(var_dir):
            print("MISSING:", var_dir)
            continue

        files = natsort.natsorted([
            f for f in os.listdir(var_dir) if f.endswith(".nc")
        ])

        print("N files:", len(files))

        for fname in files:
            path = os.path.join(var_dir, fname)
            ds = nc.Dataset(path)

            var_data = ds.variables[var_name][:].filled(np.nan)

            # ---------------------------
            # MASK (solo una volta)
            # ---------------------------
            if mask is None:
                first_layer = var_data[0]
                mask = ~np.isnan(first_layer)   # mare=True

            # ---------------------------
            # INIT COUNTER
            # ---------------------------
            if neg_counter is None:
                _, ny, nx = var_data.shape
                neg_counter = np.zeros((ny, nx))

            # ---------------------------
            # NEGATIVE MASK
            # ---------------------------
            neg_mask = (var_data < 0) & (~np.isnan(var_data))

            # accumulo su tutti i layer
            for layer in range(var_data.shape[0]):
                neg_counter += neg_mask[layer].astype(int)

            ds.close()

    # =========================
    # PLOT FINALE
    # =========================
    if neg_counter is not None and mask is not None:

        plot_map = neg_counter.copy()

        # terra = -1
        plot_map[~mask] = -1

        cmap = plt.cm.viridis.copy()
        cmap.set_under("lightgray")

        plt.figure(figsize=(8,6))

        plt.imshow(plot_map, origin="lower", cmap=cmap, vmin=0)
        plt.colorbar(label="Negative counting")

        plt.title(f"Negative counting map ({var_name})")

        plt.savefig(output_txt.replace(".txt", "_neg_map.png"))
        plt.close()

        print("Plot salvato", output_txt.replace(".txt", "_neg_map.png") )

    else:
        print("Nessun dato trovato")





if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <dir1,dir2,dir3> <var_name> <output.txt>")
        sys.exit(1)

    # lista directory
    data_dirs = sys.argv[1].split(",")
    var_name = sys.argv[2]

    output_txt = sys.argv[3] if len(sys.argv) > 3 else "dataset_stats.txt"
    # se output_txt è una directory e non un path al file txt→ crea file dentro
    if os.path.isdir(output_txt):
        output_txt = os.path.join(output_txt, f"{var_name}_stats.txt")

    # mi assicuro xhe la cartella esista
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    analyze_multi_dataset_neg(data_dirs, var_name, output_txt)
