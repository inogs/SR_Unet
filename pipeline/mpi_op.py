import json
import subprocess
import os

interp_flag = False
conversion_flag = True

path_target_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/NARF.cleanup"
# path_input_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/AdriaticNC.interp"
path_input_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/AdriaticNC"
path_output_dir = "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME"
conversion_type = "log"
number_of_processes = 16

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


list_variables = {
    "chl":"Chla",
    "no3":"N3n",
    "po4":"N1p",
    "so":"S",
    "thetao":"T"
}

# VARIABLES INTERPOLATION LAYER
if interp_flag:
    print("[Starting interpolation of variables]")
    os.makedirs(os.path.join(path_output_dir, "input", "interpolated.variables"), exist_ok=True)
    interpolation_path = os.path.join(path_output_dir, "input", "interpolated.variables")
    
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

    print("List of variables to be interpolated:")
    for var_input, var_target in list_valid_variable_pairs.items():
        print(f"    {var_input} -> {var_target}")

    for var_input, var_target in list_valid_variable_pairs.items():
        print(f"Processing variable: {var_input} -> {var_target}")

        input_folder = os.path.join(path_input_dir, var_input)
        print(f"    input_folder: {input_folder}")
        target_folder = os.path.join(path_target_dir, var_target)
        print(f"    target_folder: {target_folder}")
        output_folder = os.path.join(interpolation_path, var_input)
        print(f"    output_folder: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

        files = [f for f in os.listdir(target_folder) if f.startswith(var_target) and f.endswith(".nc")]
        if len(files) == 0:
            print(f"Error: target variable folder {target_folder} does not contain any .nc file starting with {var_target}.")
            continue
        grid_file_path = os.path.join(target_folder, files[0])
        print(f"    grid_file_path: {grid_file_path}")


        conf_interpolation = {
            "input_path": input_folder,
            "output_path": output_folder,
            "grid_file_path": grid_file_path,
            "variables": [
            {
                "name": var_input,
                "grid_variable": var_target
            }
            ]
        }

        conf_interpolation_path = os.path.join(path_output_dir, f"conf.interpolation.{var_input}.json")
        with open(conf_interpolation_path, "w") as f:
            json.dump(conf_interpolation, f, indent=4)
        print(f"    Configuration file for interpolation created for variable: {var_input}")
        print(f"    {conf_interpolation_path}")

        cmd = [
            "mpirun",
            "-np", str(number_of_processes),
            "python",
            "interpolate_bilinear.py",
            "--config", conf_interpolation_path
        ]
        subprocess.run(cmd, check=True)

        os.remove(conf_interpolation_path)
        print(f"    Configuration file for interpolation removed for variable: {var_input}")
    # end if
    print("[Interpolation of variables completed]")
# END VARIABLES INTERPOLATION LAYER

# VARIABLES CONVERSION LAYER
if conversion_flag:
    print("[Starting conversion of variables]")
    os.makedirs(os.path.join(path_output_dir, "input", "converted.variables"), exist_ok=True)
    os.makedirs(os.path.join(path_output_dir, "target", "converted.variables"), exist_ok=True)

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

    # handling interp path
    interpolation_path = os.path.join(path_output_dir, "input", "interpolated.variables")
    if not os.path.exists(interpolation_path):
        print(f"Error: interpolation path {interpolation_path} does not exist. Please run the interpolation step before the conversion step.")
        exit(1)

    for var_input, var_target in list_valid_variable_pairs.items():
        print(f"Processing variable: {var_input} -> {var_target}")

        # BEGIN: INPUT VARIABLE CONVERSION
            # input var are taken from the interpolation path
        conf_conversion = {
            "folder_path": os.path.join(interpolation_path, var_input),
            "output_path": os.path.join(path_output_dir, "input", "converted.variables", var_input + "." + conversion_type),
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
            "-np", str(number_of_processes),
            "python",
            "converter_mpi.py",
            "--config", conf_conversion_path
        ]

        subprocess.run(cmd, check=True)
        os.remove(conf_conversion_path)
        print(f"Configuration file for variable conversion removed for input variable conversion: {var_input}")
        # END

        # BEGIN: INPUT VARIABLE CONVERSION
        conf_conversion_target = {
            "folder_path": os.path.join(path_target_dir, var_target),
            "output_path": os.path.join(path_output_dir, "target", "converted.variables", var_target + "." + conversion_type),
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
            "-np", str(number_of_processes),
            "python",
            "converter_mpi.py",
            "--config", conf_conversion_target_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(conf_conversion_target_path)
        print(f"Configuration file for variable conversion removed for target variable conversion: {var_target}")
        # END
    # end for
# END VARIABLES CONVERSION LAYER
