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

######### FUNZIONE DI FEDERICA CON MODIFICHE SUL LAYER DA CONSIDERARE
# def find_best_depth_layer(var_data):
#     # trova indice del massimo assoluto
#     best_k, _, _ = np.unravel_index(np.nanargmax(var_data), var_data.shape)
#     print(f"Best depth layer: {best_k}, max value: {np.nanmax(var_data)}")
#     return best_k


# def plot_netcdf(file, var_name):
#     fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)

#     # Open the NetCDF file
#     dataset = nc.Dataset(file, 'r')

#     # Extract the variable data
#     var = dataset.variables[var_name]
#     var_data = var[:]



#     # Extract the coordinate data
#     lon = dataset.variables['longitude'][:]
#     lat = dataset.variables['latitude'][:]

#     # MODIFCIA: commento queste due righe e aggiungo quelle sotto
#     # vmin = -0.05
#     vmax = var_data.max()
#     # vmin = var_data.min()
#     # vmax = var_data.max()
    
#     # best_k = find_best_depth_layer(var_data)
#     # best_k = find_second_max_depth_layer(var_data)
#     # best_k = find_layer_highest_concentration(var_data, top_percent=10)

#     selected_layer = var_data[best_k, :, :]


#     print("Min:", np.nanmin(var_data))
#     print("Max:", np.nanmax(var_data))

#     print(f"Layer {best_k} Min:", np.nanmin(var_data[best_k,:,:]))
#     print(f"Layer {best_k} Max:", np.nanmax(var_data[best_k,:,:]))
#     num_negatives = np.sum((var_data < 0) & (~np.isnan(var_data)))
#     print("How many negatives:", num_negatives, "on:", var_data.size)
#     print("Any inf:", np.isinf(var_data).any())
#     print("Any nan:", np.isnan(var_data).any())

#     # Plot the data

#     # MODIFICA

#     # im = ax.pcolormesh(lon, lat, var_data[0, :, :], cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)

#     cmap = cmocean.cm.dense.copy()
#     cmap.set_under('red')

#     im = ax.pcolormesh(
#         lon, lat,
#         selected_layer.filled(np.nan),
#         cmap=cmap,
#         vmin=0,
#         vmax=vmax
#     )


#     # Set the labels and title
#     ax.set_xlabel('Longitudine', fontsize=40)
#     ax.set_ylabel('Latitudine', fontsize=40)
#     ax.set_title(f'{var_name} Distribution at layer {best_k}', fontsize=20)

#     ax.tick_params(axis='both', which='major', labelsize=30)

#     # Add colorbar
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(f"{var_name}", fontsize=40)
#     cbar.ax.tick_params(labelsize=30)

#     # Close the dataset
#     dataset.close()

#     # Save the plot
#     plt.savefig("single_plot.png", bbox_inches='tight')
#     plt.show()



# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python script.py <variable_name> <file>")
#         sys.exit(1)

#     var_name = sys.argv[1]
#     file = sys.argv[2]

#     plot_netcdf(file, var_name)




############# SELEZIONO IO IL LAYER DA CONSIDERARE

# def plot_netcdf(file, var_name, layer=None):
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


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python script.py <variable_name> <file> [layer_index]")
#         sys.exit(1)

#     var_name = sys.argv[1]
#     file = sys.argv[2]
#     layer = int(sys.argv[3]) if len(sys.argv) > 3 else None

#     plot_netcdf(file, var_name, layer)




############# PRINTA TUTTI I LAYER DI UN FILE .nc



# def plot_netcdf(file, var_name, output_dir):
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
#         plt.close(fig)





# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python script.py <variable_name> <file>")
#         sys.exit(1)

#     var_name = sys.argv[1]
#     file = sys.argv[2]
#     output_dir = sys.argv[3] if len(sys.argv) > 3 else f"{var_name}_layers"

#     os.makedirs(output_dir, exist_ok=True)


#     plot_netcdf(file, var_name,  output_dir)








############## FUNZIONI CON TRESHOLD

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
import os
import cmocean

def plot_netcdf(file, var_name, output_dir, threshold=None):
    # Open dataset
    dataset = nc.Dataset(file, 'r')
    var = dataset.variables[var_name]
    var_data = var[:]
    lon = dataset.variables['longitude'][:]
    lat = dataset.variables['latitude'][:]
    file_title = os.path.basename(file)

    # Statistiche globali
    # print("Global Min:", np.nanmin(var_data))
    # print("Global Max:", np.nanmax(var_data))
    # print("Any inf:", np.isinf(var_data).any())
    # print("Any nan:", np.isnan(var_data).any())

    for i, layer in enumerate(var_data):
        vmax = np.nanmax(layer)

        if var_name in ['thetao', 'so']:
            vmin = np.nanmin(layer)
        else: 
            vmin = 0 

        # print(f"\nLayer {i}")
        # print("Min:", np.nanmin(layer))
        # print("Max:", np.nanmax(layer))
        # num_negatives = np.sum((layer < 0) & (~np.isnan(layer)))
        # print("Negatives:", num_negatives, "on:", layer.size)
        
        # Calcolo percentuale sopra threshold se definito
        perc_above = None
        if threshold is not None:
            count_above = np.sum(layer > threshold)
            perc_above = 100 * count_above / layer.size
            print(f"Cells above {threshold}: {count_above} ({perc_above:.2f}%)")

        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        cmap = cmocean.cm.dense.copy()
        cmap.set_under('red')
        im = ax.pcolormesh(
            lon, lat,
            np.ma.masked_invalid(layer),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_xlabel('Longitudine', fontsize=40)
        ax.set_ylabel('Latitudine', fontsize=40)

        # Inserisco percentuale nel titolo se presente
        title = f'{var_name} - layer {i} - {file_title}'
        if perc_above is not None:
            title += f' | {perc_above:.2f}% > {threshold}'
        ax.set_title(title, fontsize=20)

        ax.tick_params(axis='both', which='major', labelsize=30)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(var_name, fontsize=40)
        cbar.ax.tick_params(labelsize=30)

        # Salvataggio plot
        plt.savefig(
            os.path.join(output_dir, f"{var_name}_layer_{i}.png"),
            bbox_inches='tight'
        )
        plt.close(fig)

    dataset.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <variable_name> <file.nc> [output_dir] [threshold_value]")
        sys.exit(1)

    var_name = sys.argv[1]
    file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else f"{var_name}_layers"
    os.makedirs(output_dir, exist_ok=True)

    # Threshold opzionale
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else None

    plot_netcdf(file, var_name, output_dir, threshold)
