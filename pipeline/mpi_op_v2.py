import json
import subprocess
import os



path_target_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/NARF.cleanup"
path_input_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/AdriaticNC"
path_preproc_dir = "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME_DEVELOP"
conversion_type = "log"
number_of_processes = 16

path_interpolation_out_dir = os.path.join(path_preproc_dir, "data.input", "interpolated")
path_interpolation_input_dir = os.path.join(path_preproc_dir, "data.input", "original")
path_interpolation_grid_ref_dir = os.path.join(path_preproc_dir, "data.target", "original")
path_log_dir = os.path.join(path_preproc_dir, "log")
path_conf_dir = os.path.join(path_preproc_dir, "conf.files")

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


def get_valid_variable_pairs(input_dir, target_dir, variables):
    list_valid_variable_pairs = {}
    for var_input, var_target in variables.items():
        input_folder = os.path.join(input_dir, var_input)
        target_folder = os.path.join(target_dir, var_target)

        if not os.path.exists(input_folder):
            print(f"Warning: input folder {input_folder} does not exist. Skipping variable pairing: {var_input} -> {var_target}")
            continue
        if not os.path.exists(target_folder):
            print(f"Warning: target folder {target_folder} does not exist. Skipping variable pairing: {var_input} -> {var_target}")
            continue
        list_valid_variable_pairs[var_input] = var_target

    return list_valid_variable_pairs


def print_variable_pairs(variable_pairs):
    print("List of variables to be interpolated:")
    for var_input, var_target in variable_pairs.items():
        print(f"    {var_input} -> {var_target}")


def find_grid_file(target_folder, var_target):
    files = sorted(
        f for f in os.listdir(target_folder)
        if f.startswith(var_target) and f.endswith(".nc")
    )

    if len(files) == 0:
        print(
            f"Error: target variable folder {target_folder} "
            f"does not contain any .nc file starting with {var_target}."
        )
        return None

    return os.path.join(target_folder, files[0])


def execute_command(cmd, log_file_path, label):
    print(f"    Running command: {' '.join(cmd)}")
    try:
        with open(log_file_path, "w") as log_file:
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"    {label} completed successfully.")
    except subprocess.CalledProcessError as exc:
        print(
            f"Warning: {label} failed with return code {exc.returncode}. "
            f"Continuing. See log file: {log_file_path}"
        )

def run_interpolation_layer():
    print("[Starting interpolation of variables]")

    list_valid_variable_pairs = get_valid_variable_pairs(
        path_interpolation_input_dir,
        path_interpolation_grid_ref_dir,
        list_variables
    )

    print_variable_pairs(list_valid_variable_pairs)

    for var_input, var_target in list_valid_variable_pairs.items():
        print(f"Processing variable: {var_input} -> {var_target}")

        input_folder = os.path.join(path_interpolation_input_dir, var_input)
        target_folder = os.path.join(path_interpolation_grid_ref_dir, var_target)
        output_folder = os.path.join(path_interpolation_out_dir, var_input)
        os.makedirs(output_folder, exist_ok=True)

        grid_file_path = find_grid_file(target_folder, var_target)
        if grid_file_path is None:
            print(f"Warning: grid file for variable {var_target} not found. Skipping interpolation for variable: {var_input}")
            continue

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

        conf_interpolation_path = os.path.join(path_conf_dir, f"conf.interpolation.{var_input}.json")
        with open(conf_interpolation_path, "w") as f:
            json.dump(conf_interpolation, f, indent=4)
        print(f"    Configuration file for interpolation created for variable: {var_input}")
        print(f"    conf_file_path: {conf_interpolation_path}")

        log_file_path = os.path.join(path_log_dir, f"interpolation_{var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = [
            "mpirun",
            "-np", str(number_of_processes),
            "python",
            "interpolate_bilinear.py",
            "--config", conf_interpolation_path
        ]
        execute_command(cmd, log_file_path, f"Interpolation for variable {var_input}")

    print("[Interpolation of variables completed]")
# end run_interpolation_layer

def run_conversion_layer():
    print("[Starting conversion of variables]")

    list_valid_variable_pairs = get_valid_variable_pairs(
        path_interpolation_input_dir,
        path_interpolation_grid_ref_dir,
        list_conversion_variables
    )

    print_variable_pairs(list_valid_variable_pairs)

    for var_input, var_target in list_valid_variable_pairs.items():
        print(f"Processing variable: {var_input} -> {var_target}")

        # var input
        conf_conversion = {
            "folder_path": os.path.join(path_preproc_dir, "data.input", "interpolated", var_input),
            "output_path": os.path.join(path_preproc_dir, "data.input", "converted", var_input + "." + conversion_type),
            "conversion_type": conversion_type,
            "variable_name": var_input
        }

        conf_conversion_path = os.path.join(path_conf_dir, f"conf.conversion.{var_input}.json")
        with open(conf_conversion_path, "w") as f:
            json.dump(conf_conversion, f, indent=4)
        print(f"    Configuration file for conversion created for variable: {var_input}")
        print(f"    conf_file_path: {conf_conversion_path}")

        cmd = [
            "mpirun",
            "-np", str(number_of_processes),
            "python",
            "converter_mpi.py",
            "--config", conf_conversion_path
        ]

        log_file_path = os.path.join(path_log_dir, f"conversion_{var_input}.log")
        print(f"    log_file_path: {log_file_path}")
        execute_command(cmd, log_file_path, f"Conversion for input variable {var_input}")


        # var target
        conf_conversion = {
            "folder_path": os.path.join(path_preproc_dir, "data.target", "original", var_target),
            "output_path": os.path.join(path_preproc_dir, "data.target", "converted", var_target + "." + conversion_type),
            "conversion_type": conversion_type,
            "variable_name": var_target
        }

        conf_conversion_path = os.path.join(path_conf_dir, f"conf.conversion.{var_target}.json")
        with open(conf_conversion_path, "w") as f:
            json.dump(conf_conversion, f, indent=4)
        print(f"    Configuration file for conversion created for variable: {var_target}")
        print(f"    conf_file_path: {conf_conversion_path}")

        cmd = [
            "mpirun",
            "-np", str(number_of_processes),
            "python",
            "converter_mpi.py",
            "--config", conf_conversion_path
        ]

        log_file_path = os.path.join(path_log_dir, f"conversion_{var_target}.log")
        print(f"    log_file_path: {log_file_path}")
        execute_command(cmd, log_file_path, f"Conversion for target variable {var_target}")


    print("[Conversion of variables completed]")
# end run_conversion_layer

if __name__ == "__main__":

    interp_flag = True
    conversion_flag = True

    if interp_flag:
        run_interpolation_layer()

    if conversion_flag:
        run_conversion_layer()
