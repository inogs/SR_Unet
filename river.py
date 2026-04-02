# script fatto da Anna per fare grafico con Y i valori dei fiumi, X il "timestamp", dove ogni linea colorata diversa è uno dei 19 fiumi -> si vede l'andamento sinusuidale (eccetto per i tre grandi fiumi ex. Po) 

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

def sort_river_files(file_list):
    pattern = re.compile(r"river_(\d{4})_(\d{3})\.txt")

    # Controlla quali file NON rispettano la regex
    bad_files = [f for f in file_list if not pattern.match(f)]
    if bad_files:
        print("Warning: i seguenti file NON rispettano la regex 'river_YYYY_NNN.txt':")
        for f in bad_files:
            print(f)

    def key(f):
        m = pattern.match(f)
        if m:
            year = int(m.group(1))
            step = int(m.group(2))
            return (year, step)
        return (9999, 9999)  # fallback per file non validi

    # Ordina i file validi e lascia quelli non validi alla fine
    return sorted(file_list, key=key)

def get_river_vector(data_path: str):
    files = os.listdir(data_path)
    files = sort_river_files(files)
    data = []

    for f in files:
        path = os.path.join(data_path, f)
        vec = np.loadtxt(path)   # 19 valori
        data.append(vec)

    return np.array(data)  # shape (timestamps, 19)

if __name__ == "__main__":

    data_path = "/leonardo_work/OGS23_PRACE_IT_0/adelsavio/rivers/vector"

    # legge i fiumi selezionati come flag, es. python script.py 11,5
    selected_rivers = None
    if len(sys.argv) > 1:
        # converte la stringa in lista di interi, e sottrae 1 per avere indici Python 0-based
        selected_rivers = [int(x)-1 for x in sys.argv[1].split(",")]

    rivers = get_river_vector(data_path)
    print("Dataset shape:", rivers.shape)

    T = rivers.shape[0]
    timestamps = np.arange(T)

    plt.figure(figsize=(10,6))

    for i in range(rivers.shape[1]):
        if selected_rivers is not None and i not in selected_rivers:
            continue
        plt.plot(timestamps, rivers[:, i], label=f"River {i+1}")

    plt.xlabel("index")
    plt.ylabel("Flow rate?")
    plt.title("River over time")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("rivers_plot.png", dpi=300)
    plt.show()
