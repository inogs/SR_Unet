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


def analyze_file(file_path, output_txt, var_name, log):
    dataset = nc.Dataset(file_path, 'r')
    var_data = dataset.variables[var_name][:].filled(np.nan) # faccio diventare le maschere nan così sono più soicura di escluderle dopo
    var_data1 = dataset.variables[var_name][:]


    log(f"\n===== FILE: {os.path.basename(file_path)} =====")

    # global stats
    global_min = np.nanmin(var_data)
    global_max = np.nanmax(var_data)
    has_nan = np.isnan(var_data1).any()
    has_inf = np.isinf(var_data).any()

    maskglobal_min = np.ma.min(var_data1)
    maskglobal_max = np.ma.max(var_data1)

    neg_mask = (var_data < 0) & (~np.isnan(var_data)) # valori negativi escludendo i NaN
    num_neg_global = np.sum(neg_mask)

    log(f"Global Min: {global_min}, { maskglobal_min}")
    log(f"Global Max: {global_max}, {maskglobal_max}")
    log(f"Any NaN: {has_nan}")
    log(f"Any Inf: {has_inf}")
    log(f"Negative cells (global): {num_neg_global}")

    # SOLO se ci sono negativi → analisi layer
    ################# MOMENTANEAMENTE CAMBIATO L'IF PER FAR SI CHE I PLOT VENGANO SEMPRE FATTI, altrimenti deve essere if num_neg_global > 0: 
    if num_neg_global >= 0:
        log("---- Layer analysis (negatives present) ----")

        for i, layer in enumerate(var_data):
            neg_layer = np.sum((layer < 0) & (~np.isnan(layer)))
            # layer_np = np.array(layer)
            # p5 = np.nanpercentile(layer_np, 5)

            log(f"Layer {i}: negatives={neg_layer}") # , p5={p5}

        # Chiamata al plot_map.py
        output_plot_dir = os.path.join(os.path.dirname(output_txt), f"{os.path.basename(file_path)}_map")
        os.makedirs(output_plot_dir, exist_ok=True)

        try:
            subprocess.run([
                "python",
                "/leonardo/home/userexternal/adelsavi/git/codice_fede/utils/plot_map.py",
                var_name,
                file_path,
                output_plot_dir
            ], check=True)
            log(f"Map saved to {output_plot_dir}")
        except subprocess.CalledProcessError as e:
            log(f"Failed to plot map for {file_path}: {e}")

    dataset.close()




def analyze_dataset(data_dir, var_name, output_txt):
    data_dir = os.path.join(data_dir, var_name)
    files = natsort.natsorted([
        f for f in os.listdir(data_dir) if f.endswith(".nc")
    ])

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log(f"Analyzing dataset: {data_dir}")
    log(f"Variable: {var_name}")
    log(f"Number of files: {len(files)}\n")

    for fname in files:
        file_path = os.path.join(data_dir, fname)
        analyze_file(file_path,output_txt, var_name, log)

    # salva tutto
    with open(output_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved stats to: {output_txt}")



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <file.nc> <var_name> [output_dir]")
        sys.exit(1)

    file_path = sys.argv[1]      # <-- singolo file
    var_name = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "plot_output"

    os.makedirs(output_dir, exist_ok=True)

    # wrapper di logging semplice
    lines = []
    def log(msg):
        print(msg)
        lines.append(msg)

    log(f"Analyzing file: {file_path}")
    log(f"Variable: {var_name}")

    analyze_file(file_path, output_dir, var_name, log)

    # salva log
    output_txt = os.path.join(output_dir, "stats.txt")
    with open(output_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved stats to: {output_txt}")

