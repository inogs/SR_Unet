import json
import subprocess
import os

path_target_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/NARF"
path_input_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/Adriatic"
path_output_dir = "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME"
conversion_type = "log"
numeber_of_processes = 16

# given the output path, check that exists, if not return error message
if not os.path.exists(path_output_dir):
    print(f"Error: output path {path_output_dir} does not exist.")
    exit(1)
# check that folders input and target exist in the output path, if not return error message
if not os.path.exists(os.path.join(path_output_dir, "input")):
    print(f"Error: input folder does not exist in the output path {path_output_dir}.")
    exit(1)
if not os.path.exists(os.path.join(path_output_dir, "target")):
    print(f"Error: target folder does not exist in the output path {path_output_dir}.")
    exit(1)


# VARIABLES CONVERSION LAYER

# create a folder named 'converted.variables' in both target and input folders, if not exist
os.makedirs(os.path.join(path_output_dir, "input", "converted.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "converted.variables"), exist_ok=True)

list_variables = {
    "chl":"Chla",
    "no3":"N3n",
    "po4":"N1p", 
    "so":"S", 
    "thetao":"T"
}

list_conversion_variables = {
    "chl":"Chla",
    "no3":"N3n",
    "po4":"N1p"
}

list_valid_variable_pairs = {}

for var_input, var_target in list_conversion_variables.items():
    input_folder = os.path.join(path_input_dir, var_input)
    target_folder = os.path.join(path_target_dir, var_target)

    if not os.path.exists(input_folder):
        print(f"Warning: input folder {input_folder} does not exist. Skipping variable pairing: {var_input} -> {var_target}")
        continue
    if not os.path.exists(target_folder):
        print(f"Warning: target folder {target_folder} does not exist. Skipping variable pairing: {var_input} -> {var_target}")
        continue
    list_valid_variable_pairs[var_input] = var_target

print("List of variables to be converted:")
for var_input, var_target in list_valid_variable_pairs.items():
    print(f"    {var_input} -> {var_target}")


for var_input, var_target in list_valid_variable_pairs.items():
    # print var input
    print(f"Processing variable: {var_input} -> {var_target}")

    conf_conversion = {
        "folder_path": os.path.join(path_input_dir, var_input),
        "output_path": os.path.join(path_output_dir, "input", "converted.variables", var_input),
        "conversion_type": conversion_type,
        "variable_name": var_input
    }

    conf_conversion_path = os.path.join(path_output_dir, f"conf.conversion.{var_input}.json")
    with open(conf_conversion_path, "w") as f:
        json.dump(conf_conversion, f, indent=4)
    print(f"Configuration file for variable conversion created for input variable conversion: {var_input}")
    print(f"    {conf_conversion_path}")

    cmd = [
        "mpirun",
        "-np", str(numeber_of_processes),
        "python",
        "converter_mpi.py",
        "--config", conf_conversion_path
    ]

    subprocess.run(cmd, check=True)


    os.remove(conf_conversion_path)
    print(f"Configuration file for variable conversion removed for input variable conversion: {var_input}")

    conf_conversion_target = {
        "folder_path": os.path.join(path_target_dir, var_target),
        "output_path": os.path.join(path_output_dir, "target", "converted.variables", var_target),
        "conversion_type": conversion_type,
        "variable_name": var_target
    }

    conf_conversion_target_path = os.path.join(path_output_dir, f"conf.conversion.{var_target}.json")
    with open(conf_conversion_target_path, "w") as f:
        json.dump(conf_conversion_target, f, indent=4)
    print(f"Configuration file for variable conversion created for target variable conversion: {var_target}")
    print(f"    {conf_conversion_target_path}")

    cmd = [
        "mpirun",
        "-np", str(numeber_of_processes),
        "python",
        "converter_mpi.py",
        "--config", conf_conversion_target_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(conf_conversion_target_path)
    print(f"Configuration file for variable conversion removed for target variable conversion: {var_target}")

# END VARIABLES CONVERSION LAYER