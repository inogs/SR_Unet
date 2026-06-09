import os
import json

# control flags
split_flag = False
stat_flag = False
make_pt_flag = True
river_flag = False

# three directories, target, input and output
path_target_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/NARF.cleanup"
path_input_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/AdriaticNC.interp"
path_output_dir = "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME_1"
path_river_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/rivers"
conversion_type = "log"

# params: train, test and validation size, seed for reproducibility, number of threads for parallel processing
test_size = 0.2
validation_size = 0.1
seed = 42
N_THREADS = 48

print(f"Target directory: {path_target_dir}")
print(f"Input directory: {path_input_dir}")
print(f"Output directory: {path_output_dir}")

# check that target and input paths exist and are directories, if not exit with error message
if not os.path.exists(path_target_dir):
    raise FileNotFoundError(f"  Target directory not found: {path_target_dir}")
if not os.path.isdir(path_target_dir):
    raise ValueError(f" Target path is not a directory: {path_target_dir}")
if not os.path.exists(path_input_dir):
    raise FileNotFoundError(f"  Input directory not found: {path_input_dir}")
if not os.path.isdir(path_input_dir):
    raise ValueError(f"  Input path is not a directory: {path_input_dir}")

# check that output path exists, if not create it
if not os.path.exists(path_output_dir):
    print(f"Output directory does not exist, creating it: {path_output_dir}")
    os.makedirs(path_output_dir)

# create subdirectories
print("Creating output subdirectories:")
print(f"    {os.path.join(path_output_dir, 'input')}")
print(f"    {os.path.join(path_output_dir, 'target')}")
print(f"    {os.path.join(path_output_dir, 'pt.files')}")
os.makedirs(os.path.join(path_output_dir, "input"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "pt.files"), exist_ok=True)


# inside both input and target subdirectories create folders split and stat
print("Creating output subdirectories:")
print(f"    {os.path.join(path_output_dir, 'input', 'split')}")
print(f"    {os.path.join(path_output_dir, 'input', 'stat')}")
print(f"    {os.path.join(path_output_dir, 'target', 'split')}")
print(f"    {os.path.join(path_output_dir, 'target', 'stat')}")
print(f"    {os.path.join(path_output_dir, 'input', 'converted.variables')}")
print(f"    {os.path.join(path_output_dir, 'target', 'converted.variables')}")
os.makedirs(os.path.join(path_output_dir, "input", "split"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "input", "stat"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "split"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "stat"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "input", "converted.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "converted.variables"), exist_ok=True)

# create subfolders for split and stat, inside both input and target, folders are named original and converted.variables
print("Creating output subdirectories:")
print(f"    {os.path.join(path_output_dir, 'input', 'split', 'original.variables')}")
print(f"    {os.path.join(path_output_dir, 'input', 'split', 'converted.variables')}")
print(f"    {os.path.join(path_output_dir, 'input', 'stat', 'original.variables')}")
print(f"    {os.path.join(path_output_dir, 'input', 'stat', 'converted.variables')}")
print(f"    {os.path.join(path_output_dir, 'target', 'split', 'original.variables')}")
print(f"    {os.path.join(path_output_dir, 'target', 'split', 'converted.variables')}")
print(f"    {os.path.join(path_output_dir, 'target', 'stat', '  original.variables')}")
print(f"    {os.path.join(path_output_dir, 'target', 'stat', 'converted.variables')}")
os.makedirs(os.path.join(path_output_dir, "input", "split", "original.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "input", "split", "converted.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "input", "stat", "original.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "input", "stat", "converted.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "split", "original.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "split", "converted.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "stat", "original.variables"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "stat", "converted.variables"), exist_ok=True)

# only for inputs, create a subfolder interpolated.variables
print(f"    {os.path.join(path_output_dir, 'input', 'interpolated.variables')}")
os.makedirs(os.path.join(path_output_dir, "input", "interpolated.variables"), exist_ok=True)

# define a path variable, it has to store the path of the interpolated variables, it will be used for the interpolation step and for the conversion step, since we want to apply the conversion after the interpolation
# path_input_interpolated_dir = os.path.join(path_output_dir, "input", "interpolated.variables")

if river_flag:
    print("Creating river subdirectories:")
   # create a rivers subdirectory inside output directory
    # create two subdirectories inside it, one for split and one for pt.files
    print(f"    {os.path.join(path_output_dir, 'rivers')}")
    print(f"    {os.path.join(path_output_dir, 'rivers', 'split')}")
    print(f"    {os.path.join(path_output_dir, 'rivers', 'pt.files')}")
    os.makedirs(os.path.join(path_output_dir, "rivers"), exist_ok=True)
    os.makedirs(os.path.join(path_output_dir, "rivers", "split"), exist_ok=True)
    os.makedirs(os.path.join(path_output_dir, "rivers", "pt.files"), exist_ok=True)

    # define a conf_split_river dictionary with the following keys: data_path, output_path, test_size, validation_size, seed
    conf_split_river = {
        "data_path": path_river_dir,
        "output_path": os.path.join(path_output_dir, "rivers", "split"),
        "test_size": test_size,
        "validation_size": validation_size,
        "seed": seed,
    }
    with open(os.path.join(path_output_dir, "conf.split.river.json"), "w") as f:
        json.dump(conf_split_river, f, indent=4)
    print(f"    {os.path.join(path_output_dir, 'conf.split.river.json')}")
    os.system(f"python split.py --config {os.path.join(path_output_dir, 'conf.split.river.json')}")
    os.remove(os.path.join(path_output_dir, "conf.split.river.json"))

    list_pt_types = ["train", "test", "val"]
    # list_pt_types = ["val"]

    for pt_type in list_pt_types:
        conf_pt_river = {
            "input_path": os.path.join(path_output_dir, "rivers", "split", f"rivers.{pt_type}.txt"),
            "output_path": os.path.join(path_output_dir, "rivers", "pt.files"),
            "label": pt_type
        }
        # print the three keys of the conf_pt_river dictionary
        print(f"Configuration for pt creation for rivers ({pt_type}):")
        print(f"    input_path: {conf_pt_river['input_path']}")
        print(f"    output_path: {conf_pt_river['output_path']}")
        print(f"    label: {conf_pt_river['label']}")

        conf_pt_river_path = os.path.join(path_output_dir, f"conf.pt.river.{pt_type}.json")
        with open(conf_pt_river_path, "w") as f:
            json.dump(conf_pt_river, f, indent=4)
        print(f"    {conf_pt_river_path}")
        print("Calling pt creation script for rivers:")
        os.system(f"python make_pt_rivers.py --config {conf_pt_river_path}")
        os.remove(conf_pt_river_path)
        # break

    

# SECTION -- SPLIT
if split_flag:

    # TARGET VARIABLES (ORIGINAL)
    conf_split_target = {
        "data_path": path_target_dir,
        "output_path": os.path.join(path_output_dir, "target", "split", "original.variables"),
        "test_size": test_size,
        "validation_size": validation_size,
        "seed": seed,
    }
    with open(os.path.join(path_output_dir, "conf.split.target.json"), "w") as f:
        json.dump(conf_split_target, f, indent=4)
    print(f"    {os.path.join(path_output_dir, 'conf.split.target.json')}")
    os.system(f"python split.py --config {os.path.join(path_output_dir, 'conf.split.target.json')}")
    os.remove(os.path.join(path_output_dir, "conf.split.target.json"))

    # INPUT VARIABLES (INTERPOLATED)
    conf_split_input = {
        "data_path": os.path.join(path_output_dir, "input", "interpolated.variables"),
        "output_path": os.path.join(path_output_dir, "input", "split", "original.variables"),
        "test_size": test_size,
        "validation_size": validation_size,
        "seed": seed,
    }
    with open(os.path.join(path_output_dir, "conf.split.input.json"), "w") as f:
        json.dump(conf_split_input, f, indent=4)    
    print(f"    {os.path.join(path_output_dir, 'conf.split.input.json')}")
    print("Calling split script for target and input:")
    os.system(f"python split.py --config {os.path.join(path_output_dir, 'conf.split.input.json')}")
    os.remove(os.path.join(path_output_dir, "conf.split.input.json"))

    # TARGET VARIABLES (CONVERTED)
    conf_split_target = {
        "data_path": os.path.join(path_output_dir,"target","converted.variables"),
        "output_path": os.path.join(path_output_dir, "target", "split", "converted.variables"),
        "test_size": test_size,
        "validation_size": validation_size,
        "seed": seed,
    }
    with open(os.path.join(path_output_dir, "conf.split.target.json"), "w") as f:
        json.dump(conf_split_target, f, indent=4)
    os.system(f"python split.py --config {os.path.join(path_output_dir, 'conf.split.target.json')}")
    os.remove(os.path.join(path_output_dir, "conf.split.target.json"))

    # INPUT VARIABLES (CONVERTED)
    conf_split_input = {
        "data_path": os.path.join(path_output_dir,"input","converted.variables"),
        "output_path": os.path.join(path_output_dir, "input", "split", "converted.variables"),
        "test_size": test_size,
        "validation_size": validation_size,
        "seed": seed,
    }
    with open(os.path.join(path_output_dir, "conf.split.input.json"), "w") as f:
        json.dump(conf_split_input, f, indent=4)
    os.system(f"python split.py --config {os.path.join(path_output_dir, 'conf.split.input.json')}")
    os.remove(os.path.join(path_output_dir, "conf.split.input.json"))
# END OF SECTION -- SPLIT


# # SECTION -- STATISTICS
if stat_flag:
    print("Creating configuration files for statistics:")

    # define json files and write them
    conf_stat_target = {
        "input_path": os.path.join(path_output_dir, "target", "split", "original.variables"),
        "output_path": os.path.join(path_output_dir, "target", "stat", "original.variables"),
        "jobs": N_THREADS,
        "recursive": False,
    }
    conf_stat_input = {
        "input_path": os.path.join(path_output_dir, "input", "split", "original.variables"),
        "output_path": os.path.join(path_output_dir, "input", "stat", "original.variables"),
        "jobs": N_THREADS,
        "recursive": False,
    }
    # add two more conf files for the converted variables
    conf_stat_target_converted = {
        "input_path": os.path.join(path_output_dir, "target", "split", "converted.variables"),
        "output_path": os.path.join(path_output_dir, "target", "stat", "converted.variables"),
        "jobs": N_THREADS,
        "recursive": False,
    }
    conf_stat_input_converted = {
        "input_path": os.path.join(path_output_dir, "input", "split", "converted.variables"),
        "output_path": os.path.join(path_output_dir, "input", "stat", "converted.variables"),
        "jobs": N_THREADS,
        "recursive": False,
    }

    with open(os.path.join(path_output_dir, "conf.stat.target.json"), "w") as f:
        json.dump(conf_stat_target, f, indent=4)
    with open(os.path.join(path_output_dir, "conf.stat.input.json"), "w") as f:
        json.dump(conf_stat_input, f, indent=4)
    with open(os.path.join(path_output_dir, "conf.stat.target.converted.json"), "w") as f:
        json.dump(conf_stat_target_converted, f, indent=4)
    with open(os.path.join(path_output_dir, "conf.stat.input.converted.json"), "w") as f:
        json.dump(conf_stat_input_converted, f, indent=4)

    print("Configuration files for statistics created:")
    print(f"    {os.path.join(path_output_dir, 'conf.stat.target.json')}")
    print(f"    {os.path.join(path_output_dir, 'conf.stat.input.json')}")
    print(f"    {os.path.join(path_output_dir, 'conf.stat.target.converted.json')}")
    print(f"    {os.path.join(path_output_dir, 'conf.stat.input.converted.json')}")

    # call stat script for both input and target
    print("Calling stat script for target and input:")
    os.system(f"python stat.py --config {os.path.join(path_output_dir, 'conf.stat.target.json')}")
    os.system(f"python stat.py --config {os.path.join(path_output_dir, 'conf.stat.input.json')}")
    os.system(f"python stat.py --config {os.path.join(path_output_dir, 'conf.stat.target.converted.json')}")
    os.system(f"python stat.py --config {os.path.join(path_output_dir, 'conf.stat.input.converted.json')}")

    # remove conf files 
    os.remove(os.path.join(path_output_dir, "conf.stat.target.json"))
    os.remove(os.path.join(path_output_dir, "conf.stat.input.json"))
    os.remove(os.path.join(path_output_dir, "conf.stat.target.converted.json"))
    os.remove(os.path.join(path_output_dir, "conf.stat.input.converted.json"))
    print("Configuration files for statistics removed.")
# # END OF SECTION -- STATISTICS


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

# SECTION -- PT CREATION
if make_pt_flag:
    # get valid pairs
    list_valid_variable_pairs = {}
    for var_input, var_target in list_variables.items():
        input_folder = os.path.join(path_input_dir, var_input)
        target_folder = os.path.join(path_target_dir, var_target)

        if not os.path.exists(input_folder):
            print(f"Warning: input folder {input_folder} does not exist. Skipping variable pairing: {var_input} -> {var_target}")
            continue
        if not os.path.exists(target_folder):
            print(f"Warning: target folder {target_folder} does not exist. Skipping variable pairing: {var_input} -> {var_target}")
            continue
        list_valid_variable_pairs[var_input] = var_target
        # append in the same way of list_variables in dictionary format, to be used for pt creation


    print("List of valid variable pairs to be converted:")
    for var_input, var_target in list_valid_variable_pairs.items():
        print(f"    {var_input} -> {var_target}")


    list_pt_types = ["train", "test", "val"]

    for var_input, var_target in list_valid_variable_pairs.items():
        for pt_type in list_pt_types:
            conf_pt = {
                "n_workers": N_THREADS,
                "path_target": os.path.join(path_output_dir, "target", "split", "original.variables", f"{var_target}.{pt_type}.txt"),
                "path_input": os.path.join(path_output_dir, "input", "split", "original.variables",f"{var_input}.{pt_type}.txt"),
                "stat_target": os.path.join(path_output_dir, "target", "stat", "original.variables",f"stat.{var_target}.train.txt"),
                "stat_input": os.path.join(path_output_dir, "input", "stat", "original.variables",f"stat.{var_input}.train.txt"),
                "var_target": var_target,
                "var_input": var_input,
                "output_path": os.path.join(path_output_dir, "pt.files"),
                "output_file_name": f"{var_input}.{var_target}.{pt_type}.dataset.pt"
            }

            output_file_path = os.path.join(conf_pt["output_path"], conf_pt["output_file_name"])
            if os.path.exists(output_file_path):
                print(f"Warning: output file {output_file_path} already exists. Skipping pt creation for variable pairing: {var_input} -> {var_target} (train)")
                continue
            
            conf_pt_path = os.path.join(path_output_dir, f"conf.pt.{var_input}_{var_target}.{pt_type}.json")
            with open(conf_pt_path, "w") as f:        json.dump(conf_pt, f, indent=4)
            print(f"Configuration file for pt creation created for variable pairing: {var_input} -> {var_target}")
            print(f"    {conf_pt_path}")

            print("Calling pt creation script:")
            # continue
            os.system(f"python make_pt.py --config {conf_pt_path}")
            os.remove(conf_pt_path)
            print("Configuration file for pt creation removed.")
        # end loop pt_type
    # end loop var_input, var_target

    # PT creation for converted variables
    for var_input, var_target in list_conversion_variables.items():
        if var_input not in list_valid_variable_pairs:
            print(f"Warning: input variable {var_input} is not a valid variable pair. Skipping converted pt creation.")
            continue

        if list_valid_variable_pairs[var_input] != var_target:
            print(f"Warning: target variable mismatch for {var_input}. Skipping converted pt creation.")
            continue

        converted_var_input = f"{var_input}.{conversion_type}"
        converted_var_target = f"{var_target}.{conversion_type}"

        for pt_type in list_pt_types:
            conf_pt = {
                "n_workers": N_THREADS,
                "path_target": os.path.join(path_output_dir, "target", "split", "converted.variables", f"{converted_var_target}.{pt_type}.txt"),
                "path_input": os.path.join(path_output_dir, "input", "split", "converted.variables", f"{converted_var_input}.{pt_type}.txt"),
                "stat_target": os.path.join(path_output_dir, "target", "stat", "converted.variables", f"stat.{converted_var_target}.train.txt"),
                "stat_input": os.path.join(path_output_dir, "input", "stat", "converted.variables", f"stat.{converted_var_input}.train.txt"),
                "var_target": var_target,
                "var_input": var_input,
                "output_path": os.path.join(path_output_dir, "pt.files"),
                "output_file_name": f"{converted_var_input}.{converted_var_target}.{pt_type}.pt"
            }

            output_file_path = os.path.join(conf_pt["output_path"], conf_pt["output_file_name"])
            if os.path.exists(output_file_path):
                print(f"Warning: output file {output_file_path} already exists. Skipping converted pt creation for variable pairing: {converted_var_input} -> {converted_var_target} ({pt_type})")
                continue

            conf_pt_path = os.path.join(path_output_dir, f"conf.pt.{converted_var_input}.{converted_var_target}.{pt_type}.json")
            with open(conf_pt_path, "w") as f:
                json.dump(conf_pt, f, indent=4)

            print(f"Configuration file for converted pt creation created for variable pairing: {converted_var_input} -> {converted_var_target}")
            print(f"    {conf_pt_path}")

            print("Calling pt creation script for converted variables:")
            os.system(f"python make_pt.py --config {conf_pt_path}")

            os.remove(conf_pt_path)
            print("Configuration file for converted pt creation removed.")


# END OF SECTION -- PT CREATION