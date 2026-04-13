# script che calcola il quinto percentile unendo i ds train test e val della variabile che gli passi per i mesi che mettendo insieme i dati per il primo layer (idem per secondo e terzo) 
# -> esempio 1 output: quinto percentile del primo layer di tutti i dati di giugno etc

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
from collections import Counter



def name_components_ave(filename):
    base = os.path.basename(filename)

    # togli estensione .nc (solo finale)
    base = base.replace(".nc", "")

    var, date = base.split("_")   # po4 , 2008-065
    year, num = date.split("-")

    return var, int(num)


def find_month(num: int):
    if num < 7:
        return "January"
    elif num < 13:
        return "February"
    elif num < 19:
        return "March"
    elif num < 25:
        return "April"
    elif num < 31:
        return "May"
    elif num < 37:
        return "June"
    elif num < 43:
        return "July"
    elif num < 49:
        return "August"
    elif num < 55:
        return "September"
    elif num < 61:
        return "October"
    elif num < 67:
        return "November"
    else:
        return "December"


def analyze_multi_dataset(data_dirs, var_name, output_txt, months_target, extra_tag):

    data = {m: {0: [], 1: [], 2: []} for m in months_target}

    for data_dir in data_dirs:
        var_dir = os.path.join(data_dir, var_name)

        print("DIR:", var_dir)

        if not os.path.exists(var_dir):
            print("MISSING:", var_dir)
            break

        files = natsort.natsorted([
            f for f in os.listdir(var_dir) if f.endswith(".nc")
        ])


        counter = Counter()



        for fname in files:
            _ , time = name_components_ave(fname)
            month = find_month(time)
            counter[month] += 1

            print('file:', fname, month)

            if month not in months_target:
                continue

            path = os.path.join(var_dir, fname)
            ds = nc.Dataset(path)

            var_data = ds.variables[var_name][:].filled(np.nan)

            for layer in [0, 1, 2]:
                data[month][layer].append(var_data[layer].flatten())

            ds.close()
        
        print(counter)


    # =========================
    # CALCOLO PERCENTILI
    # =========================
    lines = []

    for month in months_target:
        lines.append(f"\n===== {month} =====")

        for layer in [0, 1, 2]:

            if len(data[month][layer]) == 0:
                lines.append(f"Layer {layer} - NO DATA")
                continue

            all_values = np.concatenate(data[month][layer])
           #  print(len(all_values)) # 10670400: 72 (file .nc di quel mese) * lat * lon * 1(depth perchè prendo uno alla volta i primi tre layer)
            p5 = np.nanpercentile(all_values, 5)

            lines.append(f"Layer {layer} - p5: {p5}")

    # =========================
    # SAVE FILE
    # =========================

    output_dir = os.path.dirname(output_txt)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    # aggiungi var_name al filename
    base = os.path.basename(output_txt)
    name, ext = os.path.splitext(base)

    if ext == "":
        ext = ".txt"


    final_name = f"{name}_{var_name}"

    if extra_tag is not None:
        final_name += f"_{extra_tag}"

    final_output = os.path.join(output_dir, final_name + ext)

    with open(final_output, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved to {final_output}")





if __name__ == "__main__":

    data_dirs = sys.argv[1].split(",")
    var_name = sys.argv[2]
    output_txt = sys.argv[3]

    # opzionale
    extra_tag = None
    if "-altro" in sys.argv:
        idx = sys.argv.index("-altro")
        if idx + 1 < len(sys.argv):
            extra_tag = sys.argv[idx + 1]


    months_target = []
    if "-months" in sys.argv:
        idx = sys.argv.index("-months")
        if idx + 1 < len(sys.argv):
            months_target = [
                m.strip() for m in sys.argv[idx + 1].split(",")
            ]

    # DEFAULT se non passi nulla
    if months_target == []:
        months_target = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]

    analyze_multi_dataset(data_dirs, var_name, output_txt, months_target, extra_tag)
