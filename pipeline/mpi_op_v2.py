import os
import argparse
import json

from typing import Dict
from types import SimpleNamespace

import subprocess


path_conf_file = os.path.join(
    os.path.dirname(__file__),
    "conf.files.dir",
    "conf_general.json"
)

apply_treshold_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "postproc",
    "apply_treshold_mpi.py"
)

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
    print("List of valid variable pairs:")
    for var_input, var_target in variable_pairs.items():
        print(f"    {var_input} -> {var_target}")


def get_valid_treshold_variables(variable_names, thresholds):
    valid_variables = {}
    for var_name in variable_names:
        if var_name not in thresholds:
            print(f"Warning: threshold not defined for variable {var_name}. Skipping treshold for variable: {var_name}")
            continue
        valid_variables[var_name] = thresholds[var_name]

    return valid_variables


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

def run_interpolation_layer(conf):
    print("[Starting interpolation of variables]")

    path_interpolation_out_dir = os.path.join(conf.path_preproc_dir, "data.input", "interpolated")
    path_interpolation_input_dir = os.path.join(conf.path_preproc_dir, "data.input", "original")
    path_interpolation_grid_ref_dir = os.path.join(conf.path_preproc_dir, "data.target", "original")
    path_log_dir = os.path.join(conf.path_preproc_dir, "log")
    path_conf_dir = os.path.join(conf.path_preproc_dir, "conf.files")
    list_variables = vars(conf.variables)

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
            "-np", str(conf.number_of_processes),
            "python",
            "interpolate_bilinear.py",
            "--config", conf_interpolation_path
        ]
        execute_command(cmd, log_file_path, f"Interpolation for variable {var_input}")

    print("[Interpolation of variables completed]")
# end run_interpolation_layer

def run_conversion_layer(conf):
    print("[Starting conversion of variables]")

    path_interpolation_out_dir = os.path.join(conf.path_preproc_dir, "data.input", "interpolated")
    path_interpolation_grid_ref_dir = os.path.join(conf.path_preproc_dir, "data.target", "original")
    path_log_dir = os.path.join(conf.path_preproc_dir, "log")
    path_conf_dir = os.path.join(conf.path_preproc_dir, "conf.files")
    list_conversion_variables = vars(conf.conversion_variables)

    list_valid_variable_pairs = get_valid_variable_pairs(
        path_interpolation_out_dir,
        path_interpolation_grid_ref_dir,
        list_conversion_variables
    )

    print_variable_pairs(list_valid_variable_pairs)

    for var_input, var_target in list_valid_variable_pairs.items():
        print(f"Processing variable: {var_input} -> {var_target}")

        # var input
        conf_conversion = {
            "folder_path": os.path.join(conf.path_preproc_dir, "data.input", "interpolated", var_input),
            "output_path": os.path.join(conf.path_preproc_dir, "data.input", "converted", var_input + "." + conf.conversion_type),
            "conversion_type": conf.conversion_type,
            "variable_name": var_input
        }

        conf_conversion_path = os.path.join(path_conf_dir, f"conf.conversion.{var_input}.json")
        with open(conf_conversion_path, "w") as f:
            json.dump(conf_conversion, f, indent=4)
        print(f"    Configuration file for conversion created for variable: {var_input}")
        print(f"    conf_file_path: {conf_conversion_path}")

        cmd = [
            "mpirun",
            "-np", str(conf.number_of_processes),
            "python",
            "converter_mpi.py",
            "--config", conf_conversion_path
        ]

        log_file_path = os.path.join(path_log_dir, f"conversion_{var_input}.log")
        print(f"    log_file_path: {log_file_path}")
        execute_command(cmd, log_file_path, f"Conversion for input variable {var_input}")


        # var target
        conf_conversion = {
            "folder_path": os.path.join(conf.path_preproc_dir, "data.target", "original", var_target),
            "output_path": os.path.join(conf.path_preproc_dir, "data.target", "converted", var_target + "." + conf.conversion_type),
            "conversion_type": conf.conversion_type,
            "variable_name": var_target
        }

        conf_conversion_path = os.path.join(path_conf_dir, f"conf.conversion.{var_target}.json")
        with open(conf_conversion_path, "w") as f:
            json.dump(conf_conversion, f, indent=4)
        print(f"    Configuration file for conversion created for variable: {var_target}")
        print(f"    conf_file_path: {conf_conversion_path}")

        cmd = [
            "mpirun",
            "-np", str(conf.number_of_processes),
            "python",
            "converter_mpi.py",
            "--config", conf_conversion_path
        ]

        log_file_path = os.path.join(path_log_dir, f"conversion_{var_target}.log")
        print(f"    log_file_path: {log_file_path}")
        execute_command(cmd, log_file_path, f"Conversion for target variable {var_target}")


    print("[Conversion of variables completed]")
# end run_conversion_layer

def run_treshold_layer(conf):
    print("[Starting treshold application on variables]")

    path_interpolation_out_dir = os.path.join(conf.path_preproc_dir, "data.input", "interpolated")
    path_target_original_dir = os.path.join(conf.path_preproc_dir, "data.target", "original")
    path_input_treshold_out_dir = os.path.join(conf.path_preproc_dir, "data.input", "treshold")
    path_target_treshold_out_dir = os.path.join(conf.path_preproc_dir, "data.target", "treshold")
    path_log_dir = os.path.join(conf.path_preproc_dir, "log")
    path_conf_dir = os.path.join(conf.path_preproc_dir, "conf.files")
    list_treshold_variables = vars(conf.treshold_variables)
    list_thresholds = vars(conf.thresholds)

    list_valid_variable_pairs = get_valid_variable_pairs(
        path_interpolation_out_dir,
        path_target_original_dir,
        list_treshold_variables
    )

    print_variable_pairs(list_valid_variable_pairs)

    list_valid_input_thresholds = get_valid_treshold_variables(
        list_valid_variable_pairs.keys(), list_thresholds
    )
    list_valid_target_thresholds = get_valid_treshold_variables(
        list_valid_variable_pairs.values(), list_thresholds
    )

    for var_input, treshold_value in list_valid_input_thresholds.items():
        print(f"Processing input variable: {var_input}")

        input_folder = os.path.join(path_interpolation_out_dir, var_input)
        output_folder = os.path.join(path_input_treshold_out_dir, var_input)
        os.makedirs(output_folder, exist_ok=True)

        conf_treshold = {
            "folder_path": input_folder,
            "output_path": output_folder,
            "variable_name": var_input,
            "treshold": treshold_value
        }

        conf_treshold_path = os.path.join(path_conf_dir, f"conf.treshold.{var_input}.json")
        with open(conf_treshold_path, "w") as f:
            json.dump(conf_treshold, f, indent=4)
        print(f"    Configuration file for treshold created for variable: {var_input}")
        print(f"    conf_file_path: {conf_treshold_path}")

        cmd = [
            "mpirun",
            "-np", str(conf.number_of_processes),
            "python",
            apply_treshold_path,
            "--config", conf_treshold_path
        ]

        log_file_path = os.path.join(path_log_dir, f"treshold_{var_input}.log")
        print(f"    log_file_path: {log_file_path}")
        execute_command(cmd, log_file_path, f"Treshold for input variable {var_input}")

    for var_target, treshold_value in list_valid_target_thresholds.items():
        print(f"Processing target variable: {var_target}")

        target_folder = os.path.join(path_target_original_dir, var_target)
        output_folder = os.path.join(path_target_treshold_out_dir, var_target)
        os.makedirs(output_folder, exist_ok=True)

        conf_treshold = {
            "folder_path": target_folder,
            "output_path": output_folder,
            "variable_name": var_target,
            "treshold": treshold_value
        }

        conf_treshold_path = os.path.join(path_conf_dir, f"conf.treshold.{var_target}.json")
        with open(conf_treshold_path, "w") as f:
            json.dump(conf_treshold, f, indent=4)
        print(f"    Configuration file for treshold created for variable: {var_target}")
        print(f"    conf_file_path: {conf_treshold_path}")

        cmd = [
            "mpirun",
            "-np", str(conf.number_of_processes),
            "python",
            apply_treshold_path,
            "--config", conf_treshold_path
        ]

        log_file_path = os.path.join(path_log_dir, f"treshold_{var_target}.log")
        print(f"    log_file_path: {log_file_path}")
        execute_command(cmd, log_file_path, f"Treshold for target variable {var_target}")

    print("[Treshold application on variables completed]")
# end run_treshold_layer

def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))

if __name__ == "__main__":

    conf = read_conf_file(path_conf_file)

    print(f"path_target_dir: {conf.path_target_dir}")
    print(f"path_input_dir: {conf.path_input_dir}")
    print(f"path_preproc_dir: {conf.path_preproc_dir}")
    print(f"conversion_type: {conf.conversion_type}")
    print(f"number_of_processes: {conf.number_of_processes}")
    print(f"interp_flag: {conf.interp_flag}")
    print(f"conversion_flag: {conf.conversion_flag}")
    print(f"treshold_flag: {conf.treshold_flag}")
    print(f"variables: {conf.variables}")
    print(f"conversion_variables: {conf.conversion_variables}")
    print(f"treshold_variables: {conf.treshold_variables}")

    if conf.interp_flag == True:
        run_interpolation_layer(conf)

    if conf.conversion_flag == True:
        run_conversion_layer(conf)

    if conf.treshold_flag == True:
        run_treshold_layer(conf)