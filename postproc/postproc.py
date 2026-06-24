# define a print list function that prints a list of items, one per line
def print_list(lst):
    for item in lst:
        print(item)

import os

path_workdir = "/leonardo_scratch/large/userexternal/gzuccari"
path_OPA_HOME = os.path.join(path_workdir, "OPA_HOME")
path_ARCHIVE = os.path.join(path_workdir, "ARCHIVE")
path_TEST = os.path.join(path_workdir, "TEST/05.28.test.dir.LONGER")
path_predictions = os.path.join(path_workdir, "TEST/predictions")


river_path = "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/rivers/pt.files/rivers_test.pt"
print("river_path: ", river_path)

print("path_OPA_HOME: ", path_OPA_HOME)
print("path_ARCHIVE: ", path_ARCHIVE)
print("path_TEST: ", path_TEST)

# make a list of the previous 8 lines, one item each line

weights_path_list = [
    "/leonardo_scratch/large/userexternal/gzuccari/TEST/05.28.test.dir.LONGER/chl.Chla/out2/best_model_chl.Chla_191.284.ckpt",
    "/leonardo_scratch/large/userexternal/gzuccari/TEST/05.28.test.dir.LONGER/chl.log.Chla.log/out2/best_model_chl.log.Chla.log_97.188.ckpt",
    "/leonardo_scratch/large/userexternal/gzuccari/TEST/05.28.test.dir.LONGER/no3.N3n/out2/best_model_no3.N3n_98.199.ckpt",
    "/leonardo_scratch/large/userexternal/gzuccari/TEST/05.28.test.dir.LONGER/no3.log.N3n.log/out2/best_model_no3.log.N3n.log_92.192.ckpt",
    "/leonardo_scratch/large/userexternal/gzuccari/TEST/05.28.test.dir.LONGER/po4.N1p/out1/best_model_po4.N1p_100.200.ckpt",
    "/leonardo_scratch/large/userexternal/gzuccari/TEST/05.28.test.dir.LONGER/po4.log.N1p.log/out1/best_model_po4.log.N1p.log_93.200.ckpt",
    "/leonardo_scratch/large/userexternal/gzuccari/TEST/05.28.test.dir.LONGER/so.S/out1/best_model_so.S_199.281.ckpt",
    "/leonardo_scratch/large/userexternal/gzuccari/TEST/05.28.test.dir.LONGER/thetao.T/out1/best_model_thetao.T_97.200.ckpt"
]
print_list(weights_path_list)


test_pt_path_list = [
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/pt.files/chl.Chla.test.dataset.pt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/pt.files/chl.log.Chla.log.test.pt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/pt.files/no3.N3n.test.dataset.pt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/pt.files/no3.log.N3n.log.test.pt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/pt.files/po4.N1p.test.dataset.pt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/pt.files/po4.log.N1p.log.test.pt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/pt.files/so.S.test.dataset.pt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/pt.files/thetao.T.test.dataset.pt"
]
print_list(test_pt_path_list)


test_file_path_list = [
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/input/split/original.variables/chl.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/input/split/converted.variables/chl.log.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/input/split/original.variables/no3.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/input/split/converted.variables/no3.log.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/input/split/original.variables/po4.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/input/split/converted.variables/po4.log.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/input/split/original.variables/so.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/input/split/original.variables/thetao.test.txt"
]
print_list(test_file_path_list)

target_test_file_path_list = [
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/split/original.variables/Chla.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/split/converted.variables/Chla.log.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/split/original.variables/N3n.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/split/converted.variables/N3n.log.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/split/original.variables/N1p.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/split/converted.variables/N1p.log.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/split/original.variables/S.test.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/split/original.variables/T.test.txt"
]
print_list(target_test_file_path_list)

stat_file_path_list = [
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/stat/original.variables/stat.Chla.train.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/stat/converted.variables/stat.Chla.log.train.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/stat/original.variables/stat.N3n.train.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/stat/converted.variables/stat.N3n.log.train.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/stat/original.variables/stat.N1p.train.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/stat/converted.variables/stat.N1p.log.train.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/stat/original.variables/stat.S.train.txt",
    "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME/target/stat/original.variables/stat.T.train.txt"
]

print_list(stat_file_path_list)


#output path, DA FARE

variable_names_complete = [
    "chl.Chla",
    "chl.log.Chla.log",
    "no3.N3n",
    "no3.log.N3n.log",
    "po4.N1p",
    "po4.log.N1p.log",
    "so.S",
    "thetao.T"
]
print("variable_names_complete: ", variable_names_complete)

variable_names = [
    "Chla",
    "Chla",
    "N3n",
    "N3n",
    "N1p",
    "N1p",
    "S",
    "T"
]
print("variable_names: ", variable_names)

variable_names_input = [
    "chl",
    "chl",
    "no3",
    "no3",
    "po4",
    "po4",
    "so",
    "thetao"
]
print("variable_names_input: ", variable_names_input)

# using this template make a loop and print to file the following json files, one for each variable, with the corresponding paths and names:

for i in range(len(variable_names)):
    json_dict = {
        "weight_file": weights_path_list[i],
        "test_path": test_pt_path_list[i],
        "output_path": path_predictions,
        "prediction_subdir": variable_names_complete[i],
        "variables": [variable_names_input[i]],
        "reference_nc_lists": {
            variable_names_input[i]: test_file_path_list[i]
        },
        "target_variables": {
            variable_names_input[i]: variable_names[i]
        },
        "reference_target_nc_lists": {
            variable_names[i]: target_test_file_path_list[i]
        },
        "stats_files": {
            variable_names_input[i]: stat_file_path_list[i]
        },
        "river_path": river_path,
        "batch_size": 4,
        "mask_threshold": 1000000.0,
        "device": "auto",
        "seed": 0
    }
    with open(os.path.join(path_predictions, f"{variable_names_complete[i]}_postproc.json"), 'w') as f:
        import json
        json.dump(json_dict, f, indent=4)

import subprocess

for i in range(len(variable_names)):
    json_file_path = os.path.join(path_predictions, f"{variable_names_complete[i]}_postproc.json")
    subprocess.run(
        ["python3", "../produce_predictions_prod.py", "-c", json_file_path],
        check=True,
    )
