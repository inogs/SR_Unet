import os
import json

# three directories, target, input and output

path_target_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/NARF"
path_input_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/Adriatic"
path_output_dir = "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME"

# params: train, test and validation
test_size = 0.2
validation_size = 0.1
seed = 42
N_THREADS = 24

print(f"Target directory: {path_target_dir}")
print(f"Input directory: {path_input_dir}")
print(f"Output directory: {path_output_dir}")

# REMINDER: check for target and input path
# REMINDER: if output path does not exist stop and return error message

# create two subdirectories in the output
# one named 'input'
# other named 'target'
# third named 'pt.files'

# write lines to the create subdirectories
# print to terminal with indentation
print("Creating output subdirectories:")
print(f"    {os.path.join(path_output_dir, 'input')}")
print(f"    {os.path.join(path_output_dir, 'target')}")
print(f"    {os.path.join(path_output_dir, 'pt.files')}")
os.makedirs(os.path.join(path_output_dir, "input"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "pt.files"), exist_ok=True)

# inside both input and target subdirectories create subfolders named 'converted variables' and 'original variables'
# os.makedirs(os.path.join(path_output_dir, "input", "converted variables"), exist_ok=True)
# os.makedirs(os.path.join(path_output_dir, "input", "original variables"), exist_ok=True)
# os.makedirs(os.path.join(path_output_dir, "target", "converted variables"), exist_ok=True)
# os.makedirs(os.path.join(path_output_dir, "target", "original variables"), exist_ok=True)

# inside both input and target subdirectories create folders split and stat
print("Creating output subdirectories:")
print(f"    {os.path.join(path_output_dir, 'input', 'split')}")
print(f"    {os.path.join(path_output_dir, 'input', 'stat')}")
print(f"    {os.path.join(path_output_dir, 'target', 'split')}")
print(f"    {os.path.join(path_output_dir, 'target', 'stat')}")
os.makedirs(os.path.join(path_output_dir, "input", "split"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "input", "stat"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "split"), exist_ok=True)
os.makedirs(os.path.join(path_output_dir, "target", "stat"), exist_ok=True)


split_flag = True
stat_flag = True
make_pt_flag = True

# SECTION -- SPLIT
if split_flag:
    print("Creating configuration files for splitting:")
    conf_split_target = {
        "data_path": path_target_dir,
        "output_path": os.path.join(path_output_dir, "target", "split"),
        "test_size": test_size,
        "validation_size": validation_size,
        "seed": seed,
    }
    with open(os.path.join(path_output_dir, "conf.split.target.json"), "w") as f:
        json.dump(conf_split_target, f, indent=4)

    conf_split_input = {
        "data_path": path_input_dir,
        "output_path": os.path.join(path_output_dir, "input", "split"),
        "test_size": test_size,
        "validation_size": validation_size,
        "seed": seed,
    }

    with open(os.path.join(path_output_dir, "conf.split.input.json"), "w") as f:
        json.dump(conf_split_input, f, indent=4)

    print("Configuration files for splitting created:")
    print(f"    {os.path.join(path_output_dir, 'conf.split.target.json')}")
    print(f"    {os.path.join(path_output_dir, 'conf.split.input.json')}")
    # call the split script for both input and target
    print("Calling split script for target and input:")
    os.system(f"python split.py --config {os.path.join(path_output_dir, 'conf.split.target.json')}")
    os.system(f"python split.py --config {os.path.join(path_output_dir, 'conf.split.input.json')}")

    # remove json files
    os.remove(os.path.join(path_output_dir, "conf.split.target.json"))
    os.remove(os.path.join(path_output_dir, "conf.split.input.json"))
    print("Configuration files for splitting removed.")
# END OF SECTION -- SPLIT


# # SECTION -- STATISTICS
if stat_flag:
    print("Creating configuration files for statistics:")

    # define json files and write them
    conf_stat_target = {
        "input_path": os.path.join(path_output_dir, "target", "split"),
        "output_path": os.path.join(path_output_dir, "target", "stat"),
        "jobs": N_THREADS,
        "recursive": False,
    }
    conf_stat_input = {
        "input_path": os.path.join(path_output_dir, "input", "split"),
        "output_path": os.path.join(path_output_dir, "input", "stat"),
        "jobs": N_THREADS,
        "recursive": False,
    }

    with open(os.path.join(path_output_dir, "conf.stat.target.json"), "w") as f:
        json.dump(conf_stat_target, f, indent=4)
    with open(os.path.join(path_output_dir, "conf.stat.input.json"), "w") as f:
        json.dump(conf_stat_input, f, indent=4)

    print("Configuration files for statistics created:")
    print(f"    {os.path.join(path_output_dir, 'conf.stat.target.json')}")
    print(f"    {os.path.join(path_output_dir, 'conf.stat.input.json')}")

    # call stat script for both input and target
    print("Calling stat script for target and input:")
    os.system(f"python stat.py --config {os.path.join(path_output_dir, 'conf.stat.target.json')}")
    os.system(f"python stat.py --config {os.path.join(path_output_dir, 'conf.stat.input.json')}")

    # remove conf files 
    os.remove(os.path.join(path_output_dir, "conf.stat.target.json"))
    os.remove(os.path.join(path_output_dir, "conf.stat.input.json"))
    print("Configuration files for statistics removed.")
# # END OF SECTION -- STATISTICS


list_variables = {
    "chl":"Chla",
    "no3":"N3n",
    "po4":"N1p", 
    "so":"S", 
    "thetao":"T"
}

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
    # list_pt_types = ["train"]
    # SECTION -- PT CREATION
    for var_input, var_target in list_valid_variable_pairs.items():
        for pt_type in list_pt_types:
            conf_pt = {
                "n_workers": N_THREADS,
                "path_target": os.path.join(path_output_dir, "target", "split", f"{var_target}.{pt_type}.txt"),
                "path_input": os.path.join(path_output_dir, "input", "split", f"{var_input}.{pt_type}.txt"),
                "stat_target": os.path.join(path_output_dir, "target", "stat", f"stat.{var_target}.train.txt"),
                "stat_input": os.path.join(path_output_dir, "input", "stat", f"stat.{var_input}.train.txt"),
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

