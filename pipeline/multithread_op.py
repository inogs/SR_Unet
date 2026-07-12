import os
import argparse
import json
import shlex

from typing import Dict
from types import SimpleNamespace

path_conf_file = os.path.join(
    os.path.dirname(__file__),
    "conf.files.dir",
    "conf_general.json"
)


def execute_command(cmd, log_file_path, label):
    command = " ".join(shlex.quote(str(part)) for part in cmd)
    command_with_log = f"{command} > {shlex.quote(log_file_path)} 2>&1"

    print(f"    Running command: {command}")
    return_code = os.system(command_with_log)

    if return_code == 0:
        print(f"    {label} completed successfully.")
        return True

    exit_code = os.waitstatus_to_exitcode(return_code)
    print(
        f"Warning: {label} failed with return code {exit_code}. "
        f"Continuing. See log file: {log_file_path}"
    )
    return False

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
    print(f"variables: {conf.variables}")
    print(f"conversion_variables: {conf.conversion_variables}")
    print(f"split: {conf.split}")

    split_label = conf.split.label

    os.makedirs(os.path.join(conf.path_preproc_dir, "splits"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label), exist_ok=True)

    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "interpolated"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "original"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "rivers"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "pt.files"), exist_ok=True)

    path_conf_dir = os.path.join(conf.path_preproc_dir, "conf.files")
    split_script_path = os.path.join(os.path.dirname(__file__), "split_v2.py")
    path_log_dir = os.path.join(conf.path_preproc_dir, "log")
    path_split_input_interpolated_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "interpolated")
    path_split_input_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "converted")
    path_split_target_original_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "original")
    path_split_target_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "converted")
    path_interpolated_input_dir = os.path.join(conf.path_preproc_dir, "data.input", "interpolated")
    path_original_target_dir = os.path.join(conf.path_preproc_dir, "data.target", "original")
    path_converted_input_dir = os.path.join(conf.path_preproc_dir, "data.input", "converted")
    path_converted_target_dir = os.path.join(conf.path_preproc_dir, "data.target", "converted")

    list_variables = vars(conf.variables)
    print(list_variables)

    for var_input, var_target in list_variables.items():
        print(f"    var_input: {var_input}")
        print(f"    var_target: {var_target}")

        conf_split = {
            "data_path": os.path.join(path_interpolated_input_dir, var_input),
            "output_path": path_split_input_interpolated_dir,
            "label": var_input,
            "test_size": conf.split.test_size,
            "validation_size": conf.split.validation_size,
            "seed": conf.split.seed,
        }

        conf_split_path = os.path.join(path_conf_dir, f"conf.split.{var_input}.json")
        with open(conf_split_path, "w") as f:
            json.dump(conf_split, f, indent=4)
        print(f"    Configuration file for split created for variable: {var_input}")
        print(f"    conf_file_path: {conf_split_path}")

        log_file_path = os.path.join(path_log_dir, f"split_{var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", split_script_path, "--config", conf_split_path]
        execute_command(cmd, log_file_path, f"Split for input variable {var_input}")

        conf_split = {
            "data_path": os.path.join(path_original_target_dir, var_target),
            "output_path": path_split_target_original_dir,
            "label": var_target,
            "test_size": conf.split.test_size,
            "validation_size": conf.split.validation_size,
            "seed": conf.split.seed,
        }

        conf_split_path = os.path.join(path_conf_dir, f"conf.split.{var_target}.json")
        with open(conf_split_path, "w") as f:
            json.dump(conf_split, f, indent=4)
        print(f"    Configuration file for split created for target variable: {var_target}")
        print(f"    conf_file_path: {conf_split_path}")

        log_file_path = os.path.join(path_log_dir, f"split_{var_target}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", split_script_path, "--config", conf_split_path]
        execute_command(cmd, log_file_path, f"Split for target variable {var_target}")


    list_conversion_variables = vars(conf.conversion_variables)
    print(list_conversion_variables)

    for var_input, var_target in list_conversion_variables.items():
        converted_var_input = f"{var_input}.{conf.conversion_type}"
        converted_var_target = f"{var_target}.{conf.conversion_type}"

        print(f"    converted_var_input: {converted_var_input}")
        print(f"    converted_var_target: {converted_var_target}")

        conf_split = {
            "data_path": os.path.join(path_converted_input_dir, converted_var_input),
            "output_path": path_split_input_converted_dir,
            "label": converted_var_input,
            "test_size": conf.split.test_size,
            "validation_size": conf.split.validation_size,
            "seed": conf.split.seed,
        }

        conf_split_path = os.path.join(path_conf_dir, f"conf.split.{converted_var_input}.json")
        with open(conf_split_path, "w") as f:
            json.dump(conf_split, f, indent=4)
        print(f"    Configuration file for split created for converted input variable: {converted_var_input}")
        print(f"    conf_file_path: {conf_split_path}")

        log_file_path = os.path.join(path_log_dir, f"split_{converted_var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", split_script_path, "--config", conf_split_path]
        execute_command(cmd, log_file_path, f"Split for converted input variable {converted_var_input}")

        conf_split = {
            "data_path": os.path.join(path_converted_target_dir, converted_var_target),
            "output_path": path_split_target_converted_dir,
            "label": converted_var_target,
            "test_size": conf.split.test_size,
            "validation_size": conf.split.validation_size,
            "seed": conf.split.seed,
        }

        conf_split_path = os.path.join(path_conf_dir, f"conf.split.{converted_var_target}.json")
        with open(conf_split_path, "w") as f:
            json.dump(conf_split, f, indent=4)
        print(f"    Configuration file for split created for converted target variable: {converted_var_target}")
        print(f"    conf_file_path: {conf_split_path}")

        log_file_path = os.path.join(path_log_dir, f"split_{converted_var_target}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", split_script_path, "--config", conf_split_path]
        execute_command(cmd, log_file_path, f"Split for converted target variable {converted_var_target}")
