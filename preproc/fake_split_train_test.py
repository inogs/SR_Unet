
#   QUESTO CODICE NON SPOSTA VERAMENTE I FILE MA USA LA STESSA LOGICA DELLO SCRIPT SPLIT_TRAIN_TEST E STAMPA ALCUNE INFORMAZIONI SUI NUMERO DI FILE CHE VENGONO EFFETTIVMENTE SPOSTATI, NUMERO FILE PER OGNI STAGIONE ETC.

import os
import random
import sys
import json
from typing import List, Any
from alive_progress import alive_bar

seed = 42

def confirm_split(files_to_move):
    print("\n==============================")
    print("ATTENZIONE: stai per splittare il dataset.")
    print(f"Numero di timestamp selezionati: {len(files_to_move)}")
    print("I file verranno SPOSTATI")
    print("==============================")

    answer = input("Sei sicuro di voler continuare? (yes/no): ")

    if answer.lower() not in ["yes", "y"]:
        print("Operazione annullata.")
        sys.exit(0)

    print("Conferma ricevuta. Procedo con lo split.\n")


def print_files_per_season(source_dir: str, var_name: str, is_cms: bool) -> None:
    """Stampa il numero di file per stagione per una variabile"""
    file_list = os.listdir(os.path.join(source_dir, var_name))
    
    seasons = {"winter": [], "spring": [], "summer": [], "autumn": []}
    for file_name in file_list:
        # Estrai l’indice temporale dai nomi dei file
        t = int(file_name[-6:-3])
        if 0 <= t < 18:
            seasons["winter"].append(file_name)
        elif 18 <= t < 36:
            seasons["spring"].append(file_name)
        elif 36 <= t < 54:
            seasons["summer"].append(file_name)
        else:
            seasons["autumn"].append(file_name)

    print(f"\nVariabile '{var_name}': {len(file_list)} file totali")
    for ssn, l in seasons.items():
        print(f"  Stagione {ssn}: {len(l)} file")


def get_random_files(source_dir: str, percentage: float, reference_var: str) -> List[str]:
    random.seed(seed)
    source_dir = os.path.join(source_dir, reference_var)
    file_list = os.listdir(source_dir)
    print(f"Dataset iniziale per '{reference_var}': {len(file_list)} file totali")

    # Divide files in 4 seasons
    seasons = {"winter": [], "spring": [], "summer": [], "autumn": []}
    for file_name in file_list:
        t = int(file_name[-6:-3])
        if 0 <= t < 18:
            seasons["winter"].append(file_name)
        elif 18 <= t < 36:
            seasons["spring"].append(file_name)
        elif 36 <= t < 54:
            seasons["summer"].append(file_name)
        else:
            seasons["autumn"].append(file_name)

    # Mostra il numero di file per stagione
    for ssn, l in seasons.items():
        print(f"Stagione {ssn}: {len(l)} file")

    # Calcola i file da muovere
    files_to_move = []
    for ssn, l in seasons.items():
        num_files = min(len(l), max(1, int(len(l) * percentage)))
        random_files = random.sample(l, num_files) if l else []
        files_to_move += random_files
        print(f"Selezionati {len(random_files)} file dalla stagione {ssn} (int(len*%))")

    print(f"Totale file selezionati per validation/test: {len(files_to_move)}\n")
    return files_to_move

def move_files(files_to_move: List[str], source_dir: str, dest_dir: str, var_name: str, is_cms: bool, bar: Any) -> None:
    """
    Simula lo spostamento dei file, conta quanti sono presenti
    e stampa quelli mancanti solo per i river.
    """
    file_list = os.listdir(source_dir)
    count_moved = 0
    missing_files = []

    for file in files_to_move:
        # Genera il nome del file nel dataset
        name_end = file[4:] if is_cms else file[4:].replace("-", "_")
        if var_name == "river":
            name_end = name_end.replace("nc", "txt")
        reference_name = var_name + "_" + name_end

        if reference_name in file_list:
            count_moved += 1
            bar()
        else:
            if var_name == "river":
                missing_files.append(reference_name)

    print(f"Totale file spostati (simulati) per {var_name}: {count_moved}")
    if var_name == "river":
        print(f"File river mancanti nel dataset river: {len(missing_files)}")
        for f in missing_files:
            print(f"  {f}")
    print()

def extract_timestamp(filename: str, var_type: str) -> str:
    """Estrae il timestamp completo dai nomi dei file"""
    if var_type == "river":
        ts = filename.replace("river_", "").replace(".txt", "")
    else:
        ts = filename.split("_", 1)[1].replace(".nc", "")
    return ts  # <-- restituisce tutto il timestamp


if __name__ == '__main__':
    i = 1
    test_size = None
    data_path = None
    val = False

    while i < len(sys.argv):
        if sys.argv[i] == "-dp":
            data_path = sys.argv[i+1]; i += 2
        elif sys.argv[i] == "-ts":
            test_size = sys.argv[i+1]; i += 2
        elif sys.argv[i] == "-val":
            val = True; i += 1
        else:
            i += 1

    if data_path is None: raise TypeError("Missing value for data path")
    if test_size is None: raise TypeError("You need to specify the size of the test set")

    map_path = os.path.join(data_path, "cms2ogs.json")
    with open(map_path, 'r') as f:
        cms2ogs_map = json.load(f)

    var_list = list(cms2ogs_map.keys())

    cms_path_3D = os.path.join(data_path, "iCMS_nc")
    ogs_path_3D = os.path.join(data_path, "NARF_nc")
    if val:
        cms_test_path_3D = os.path.join(data_path, "nc_iCMS_val")
        ogs_test_path_3D = os.path.join(data_path, "nc_OGS_val")

    percentage_to_move = float(test_size)
    reference_path = cms_path_3D
    files_to_move = get_random_files(reference_path, percentage_to_move, var_list[0])
    
    confirm_split(files_to_move)
    
    print(f"[split_test_train with test size {test_size}] Starting execution\n")
    with alive_bar(0, title="Simulazione spostamento file...") as bar:
        for var in var_list:
            source_dir_cms = os.path.join(cms_path_3D, var)
            source_dir_ogs = os.path.join(ogs_path_3D, cms2ogs_map[var])
            dest_dir_cms = os.path.join(cms_test_path_3D, var)
            dest_dir_ogs = os.path.join(ogs_test_path_3D, cms2ogs_map[var])

            print_files_per_season(cms_path_3D, var, is_cms=True)
            # print_files_per_season(ogs_path_3D, var, is_cms=True)

            move_files(files_to_move, source_dir_cms, dest_dir_cms, var, True, bar)
            move_files(files_to_move, source_dir_ogs, dest_dir_ogs, cms2ogs_map[var], False, bar)

        # Rivers
    # Rivers
    if val:
        source_dir_riv = os.path.join(data_path, "rivers", "rivers_train")
        dest_dir_riv = os.path.join(data_path, "rivers", "vector_val")
    else:
        source_dir_riv = os.path.join(data_path, "rivers", "vector")
        dest_dir_riv = os.path.join(data_path, "rivers", "vector_test")
    move_files(files_to_move, source_dir_riv, dest_dir_riv, "river", False, bar)
    # print(files_to_move)
    # Controllo timestamp river mancanti solo per la variabile di riferimento
    river_files = os.listdir(source_dir_riv)
    river_timestamps = set(extract_timestamp(f, "river") for f in river_files)

    # Trova i file della variabile di riferimento che non hanno corrispondenza nel river
    missing_files = [f for f in files_to_move if extract_timestamp(f, "var") not in river_timestamps]

    # if missing_files:
    #     print(f"Attenzione! File della variabile '{var_list[0]}' assenti nei river:")
    #     for f in missing_files:
    #         print(f"  {f}")
    # else:
    #     print(f"Tutti i file della variabile '{var_list[0]}' presenti nei river")

