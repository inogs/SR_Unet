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





############# SELEZIONO IO IL LAYER DA CONSIDERARE

# def plot_choose_the_layer_netcdf(file, var_name, layer=None):
#     fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)

#     dataset = nc.Dataset(file, 'r')
#     var = dataset.variables[var_name]
#     var_data = var[:]
#     lon = dataset.variables['longitude'][:]
#     lat = dataset.variables['latitude'][:]

#     # Scegli layer
#     if layer is None:
#         best_k = np.unravel_index(np.nanargmax(var_data), var_data.shape)[0]
#         print(f"No layer specified. Using layer with max value: {best_k}")
#     else:
#         best_k = layer
#         if best_k < 0 or best_k >= var_data.shape[0]:
#             raise ValueError(f"Layer {best_k} out of bounds. Must be between 0 and {var_data.shape[0]-1}")
#         print(f"Using specified layer: {best_k}")

#     selected_layer = var_data[best_k, :, :]

#     print("Global Min:", np.nanmin(var_data))
#     print("Global Max:", np.nanmax(var_data))
#     print(f"Layer {best_k} Min:", np.nanmin(selected_layer))
#     print(f"Layer {best_k} Max:", np.nanmax(selected_layer))

#     # Plot
#     vmax = np.nanmax(var_data)
#     cmap = cmocean.cm.dense.copy()
#     cmap.set_under('red')

#     im = ax.pcolormesh(
#         lon, lat,
#         np.ma.masked_invalid(selected_layer),
#         cmap=cmap,
#         vmin=0,
#         vmax= np.nanmax(selected_layer)
#     )

#     ax.set_xlabel('Longitudine', fontsize=40)
#     ax.set_ylabel('Latitudine', fontsize=40)
#     ax.set_title(f'{var_name} Distribution at layer {best_k}', fontsize=20)
#     ax.tick_params(axis='both', which='major', labelsize=30)

#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(f"{var_name}", fontsize=40)
#     cbar.ax.tick_params(labelsize=30)

#     dataset.close()
#     plt.savefig(f"{var_name}_layer_{best_k}.png", bbox_inches='tight')
#     plt.show()






# ############# PRINTA TUTTI I LAYER DI UN FILE .nc



# def plot_all_layers_netcdf(file, var_name, output_dir):
#     fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)

#     # Open the NetCDF file
#     dataset = nc.Dataset(file, 'r')

#     # Extract the variable data
#     var = dataset.variables[var_name]
#     var_data = var[:]



#     # Extract the coordinate data
#     lon = dataset.variables['longitude'][:]
#     lat = dataset.variables['latitude'][:]



#     # statistiche globali (fuori dal loop)
#     print("Global Min:", np.nanmin(var_data))
#     print("Global Max:", np.nanmax(var_data))
#     print("Any inf:", np.isinf(var_data).any())
#     print("Any nan:", np.isnan(var_data).any())



#     for i, layer in enumerate(var_data):
#         vmax = np.nanmax(layer)

#         print(f"\nLayer {i}")
#         print("Min:", np.nanmin(layer))
#         print("Max:", np.nanmax(layer))

#         num_negatives = np.sum((layer < 0) & (~np.isnan(layer)))
#         print("Negatives:", num_negatives, "on:", layer.size)

#         # nuova figura ad ogni iterazione
#         fig, ax = plt.subplots(figsize=(16, 8))

#         cmap = cmocean.cm.dense.copy()
#         cmap.set_under('red')

#         im = ax.pcolormesh(
#             lon, lat,
#             np.ma.masked_invalid(layer),
#             cmap=cmap,
#             vmin=0,
#             vmax=vmax
#         )

#         ax.set_xlabel('Longitudine', fontsize=40)
#         ax.set_ylabel('Latitudine', fontsize=40)
#         ax.set_title(f'{var_name} - layer {i}', fontsize=20)
#         ax.tick_params(axis='both', which='major', labelsize=30)

#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label(var_name, fontsize=40)
#         cbar.ax.tick_params(labelsize=30)


#         plt.savefig(
#             os.path.join(output_dir, f"{var_name}_layer_{i}.png"),
#             bbox_inches='tight'
#         )
# #         plt.close(fig)









# ############## FUNZIONI CON TRESHOLD


# def plot_treshold_netcdf(file, var_name, output_dir, threshold=None):
#     # Open dataset
#     dataset = nc.Dataset(file, 'r')
#     var = dataset.variables[var_name]
#     var_data = var[:]
#     lon = dataset.variables['longitude'][:]
#     lat = dataset.variables['latitude'][:]

#     # Statistiche globali
#     print("Global Min:", np.nanmin(var_data))
#     print("Global Max:", np.nanmax(var_data))
#     print("Any inf:", np.isinf(var_data).any())
#     print("Any nan:", np.isnan(var_data).any())
#     num_neg_global = np.sum((var_data < 0) & (~np.isnan(var_data)))
#     print("Number of negative cells (global):", num_neg_global)

#     for i, layer in enumerate(var_data):
#         vmax = np.nanmax(layer)

#         print(f"\nLayer {i}")
#         print("Min:", np.nanmin(layer))
#         print("Max:", np.nanmax(layer))
#         num_negatives = np.sum((layer < 0) & (~np.isnan(layer)))
#         print("Negatives:", num_negatives, "on:", layer.size)
#         # 5 percentile
#         p5 = np.nanpercentile(layer, 5)
#         print("5th percentile:", p5)


#         # Calcolo percentuale sopra threshold se definito
#         perc_above = None
#         if threshold is not None:
#             count_above = np.sum(layer > threshold)
#             perc_above = 100 * count_above / layer.size
#             print(f"Cells above {threshold}: {count_above} ({perc_above:.2f}%)")

#         # Plot
#         fig, ax = plt.subplots(figsize=(16, 8))
#         cmap = cmocean.cm.dense.copy()
#         cmap.set_under('red')
#         im = ax.pcolormesh(
#             lon, lat,
#             np.ma.masked_invalid(layer),
#             cmap=cmap,
#             vmin=0,
#             vmax=vmax
#         )

#         ax.set_xlabel('Longitudine', fontsize=40)
#         ax.set_ylabel('Latitudine', fontsize=40)

#         # Inserisco percentuale nel titolo se presente
#         title = f'{var_name} - layer {i}'
#         if perc_above is not None:
#             title += f' | {perc_above:.2f}% > {threshold}'
#         ax.set_title(title, fontsize=20)

#         ax.tick_params(axis='both', which='major', labelsize=30)
#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label(var_name, fontsize=40)
#         cbar.ax.tick_params(labelsize=30)

#         # Salvataggio plot
#         plt.savefig(
#             os.path.join(output_dir, f"{var_name}_layer_{i}.png"),
#             bbox_inches='tight'
#         )
#         plt.close(fig)

#     dataset.close()


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python script.py <variable_name> <file.nc> [output_dir] [threshold_value]")
#         sys.exit(1)

#     var_name = sys.argv[1]
#     file = sys.argv[2]
#     output_dir = sys.argv[3] if len(sys.argv) > 3 else f"{var_name}_layers"
#     os.makedirs(output_dir, exist_ok=True)

#     # Threshold opzionale
#     threshold = float(sys.argv[4]) if len(sys.argv) > 4 else None

#     plot_treshold_netcdf(file, var_name, output_dir, threshold)




def analyze_file(file_path, output_txt, var_name, log):
    # file_path = os.path.join(file_path, var_name)
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
    if num_neg_global > 0:
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
        print("Usage: python script.py <data_dir> <var_name> [output.txt]")
        sys.exit(1)

    data_dir = sys.argv[1]
    var_name = sys.argv[2]

    output_txt = sys.argv[3] if len(sys.argv) > 3 else "dataset_stats.txt"
    # se è directory → crea file dentro
    if os.path.isdir(output_txt):
        output_txt = os.path.join(output_txt, f"{var_name}_stats.txt")

    # assicurati che la cartella esista
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    analyze_dataset(data_dir, var_name, output_txt)
