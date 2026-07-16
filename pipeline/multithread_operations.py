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


def run_split_layer(conf):
    split_label = conf.split.label

    os.makedirs(os.path.join(conf.path_preproc_dir, "splits"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label), exist_ok=True)

    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "interpolated"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "treshold"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "treshold.converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "original"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "treshold"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "treshold.converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "rivers"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "pt.files"), exist_ok=True)

    path_conf_dir = os.path.join(conf.path_preproc_dir, "conf.files")
    split_script_path = os.path.join(os.path.dirname(__file__), "split.py")
    path_log_dir = os.path.join(conf.path_preproc_dir, "log")
    path_split_input_interpolated_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "interpolated")
    path_split_input_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "converted")
    path_split_input_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "treshold")
    path_split_input_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "treshold.converted")
    path_split_target_original_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "original")
    path_split_target_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "converted")
    path_split_target_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "treshold")
    path_split_target_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "treshold.converted")
    path_interpolated_input_dir = os.path.join(conf.path_preproc_dir, "data.input", "interpolated")
    path_original_target_dir = os.path.join(conf.path_preproc_dir, "data.target", "original")
    path_converted_input_dir = os.path.join(conf.path_preproc_dir, "data.input", "converted")
    path_converted_target_dir = os.path.join(conf.path_preproc_dir, "data.target", "converted")
    path_treshold_input_dir = os.path.join(conf.path_preproc_dir, "data.input", "treshold")
    path_treshold_target_dir = os.path.join(conf.path_preproc_dir, "data.target", "treshold")
    path_treshold_converted_input_dir = os.path.join(conf.path_preproc_dir, "data.input", "treshold.converted")
    path_treshold_converted_target_dir = os.path.join(conf.path_preproc_dir, "data.target", "treshold.converted")
    path_split_rivers_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "rivers")
    path_rivers_dir = os.path.join(conf.path_preproc_dir, "data.rivers", "rivers")

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


    list_treshold_variables = vars(conf.treshold_variables)
    print(list_treshold_variables)

    for var_input, var_target in list_treshold_variables.items():
        print(f"    var_input: {var_input}")
        print(f"    var_target: {var_target}")

        conf_split = {
            "data_path": os.path.join(path_treshold_input_dir, var_input),
            "output_path": path_split_input_treshold_dir,
            "label": var_input,
            "test_size": conf.split.test_size,
            "validation_size": conf.split.validation_size,
            "seed": conf.split.seed,
        }

        conf_split_path = os.path.join(path_conf_dir, f"conf.split.treshold.{var_input}.json")
        with open(conf_split_path, "w") as f:
            json.dump(conf_split, f, indent=4)
        print(f"    Configuration file for split created for tresholded input variable: {var_input}")
        print(f"    conf_file_path: {conf_split_path}")

        log_file_path = os.path.join(path_log_dir, f"split_treshold_{var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", split_script_path, "--config", conf_split_path]
        execute_command(cmd, log_file_path, f"Split for tresholded input variable {var_input}")

        conf_split = {
            "data_path": os.path.join(path_treshold_target_dir, var_target),
            "output_path": path_split_target_treshold_dir,
            "label": var_target,
            "test_size": conf.split.test_size,
            "validation_size": conf.split.validation_size,
            "seed": conf.split.seed,
        }

        conf_split_path = os.path.join(path_conf_dir, f"conf.split.treshold.{var_target}.json")
        with open(conf_split_path, "w") as f:
            json.dump(conf_split, f, indent=4)
        print(f"    Configuration file for split created for tresholded target variable: {var_target}")
        print(f"    conf_file_path: {conf_split_path}")

        log_file_path = os.path.join(path_log_dir, f"split_treshold_{var_target}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", split_script_path, "--config", conf_split_path]
        execute_command(cmd, log_file_path, f"Split for tresholded target variable {var_target}")


    list_treshold_conversion_variables = vars(conf.treshold_variables)
    print(list_treshold_conversion_variables)

    for var_input, var_target in list_treshold_conversion_variables.items():
        converted_var_input = f"{var_input}.{conf.conversion_type}"
        converted_var_target = f"{var_target}.{conf.conversion_type}"

        print(f"    converted_var_input: {converted_var_input}")
        print(f"    converted_var_target: {converted_var_target}")

        conf_split = {
            "data_path": os.path.join(path_treshold_converted_input_dir, converted_var_input),
            "output_path": path_split_input_treshold_converted_dir,
            "label": converted_var_input,
            "test_size": conf.split.test_size,
            "validation_size": conf.split.validation_size,
            "seed": conf.split.seed,
        }

        conf_split_path = os.path.join(path_conf_dir, f"conf.split.treshold.{converted_var_input}.json")
        with open(conf_split_path, "w") as f:
            json.dump(conf_split, f, indent=4)
        print(f"    Configuration file for split created for tresholded converted input variable: {converted_var_input}")
        print(f"    conf_file_path: {conf_split_path}")

        log_file_path = os.path.join(path_log_dir, f"split_treshold_{converted_var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", split_script_path, "--config", conf_split_path]
        execute_command(cmd, log_file_path, f"Split for tresholded converted input variable {converted_var_input}")

        conf_split = {
            "data_path": os.path.join(path_treshold_converted_target_dir, converted_var_target),
            "output_path": path_split_target_treshold_converted_dir,
            "label": converted_var_target,
            "test_size": conf.split.test_size,
            "validation_size": conf.split.validation_size,
            "seed": conf.split.seed,
        }

        conf_split_path = os.path.join(path_conf_dir, f"conf.split.treshold.{converted_var_target}.json")
        with open(conf_split_path, "w") as f:
            json.dump(conf_split, f, indent=4)
        print(f"    Configuration file for split created for tresholded converted target variable: {converted_var_target}")
        print(f"    conf_file_path: {conf_split_path}")

        log_file_path = os.path.join(path_log_dir, f"split_treshold_{converted_var_target}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", split_script_path, "--config", conf_split_path]
        execute_command(cmd, log_file_path, f"Split for tresholded converted target variable {converted_var_target}")


    conf_split = {
        "data_path": path_rivers_dir,
        "output_path": path_split_rivers_dir,
        "label": "rivers",
        "test_size": conf.split.test_size,
        "validation_size": conf.split.validation_size,
        "seed": conf.split.seed,
    }

    conf_split_path = os.path.join(path_conf_dir, "conf.split.rivers.json")
    with open(conf_split_path, "w") as f:
        json.dump(conf_split, f, indent=4)
    print(f"    Configuration file for split created for rivers")
    print(f"    conf_file_path: {conf_split_path}")

    log_file_path = os.path.join(path_log_dir, "split_rivers.log")
    print(f"    log_file_path: {log_file_path}")

    cmd = ["python", split_script_path, "--config", conf_split_path]
    execute_command(cmd, log_file_path, "Split for rivers")
# end run_split_layer

def compute_statistics(conf):
    split_label = conf.split.label

    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "interpolated"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "treshold"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "treshold.converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "original"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "converted"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "treshold"), exist_ok=True)
    os.makedirs(os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "treshold.converted"), exist_ok=True)

    path_conf_dir = os.path.join(conf.path_preproc_dir, "conf.files")
    stat_script_path = os.path.join(os.path.dirname(__file__), "stat.py")
    path_log_dir = os.path.join(conf.path_preproc_dir, "log")

    path_split_input_interpolated_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "interpolated")
    path_split_input_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "converted")
    path_split_input_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "treshold")
    path_split_input_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "treshold.converted")
    path_split_target_original_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "original")
    path_split_target_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "converted")
    path_split_target_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "treshold")
    path_split_target_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "treshold.converted")

    path_stat_input_interpolated_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "interpolated")
    path_stat_input_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "converted")
    path_stat_input_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "treshold")
    path_stat_input_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "treshold.converted")
    path_stat_target_original_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "original")
    path_stat_target_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "converted")
    path_stat_target_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "treshold")
    path_stat_target_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "treshold.converted")

    list_variables = vars(conf.variables)
    print(list_variables)

    for var_input, var_target in list_variables.items():
        print(f"    var_input: {var_input}")
        print(f"    var_target: {var_target}")

        conf_stat = {
            "input_path": os.path.join(path_split_input_interpolated_dir, f"{var_input}.train.txt"),
            "output_path": path_stat_input_interpolated_dir,
            "number_of_threads": conf.multithread_operations.number_of_threads,
            "var_name": var_input,
        }

        conf_stat_path = os.path.join(path_conf_dir, f"conf.stat.{var_input}.json")
        with open(conf_stat_path, "w") as f:
            json.dump(conf_stat, f, indent=4)
        print(f"    Configuration file for stat created for input variable: {var_input}")
        print(f"    conf_file_path: {conf_stat_path}")

        log_file_path = os.path.join(path_log_dir, f"stat_{var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", stat_script_path, "--config", conf_stat_path]
        execute_command(cmd, log_file_path, f"Stat for input variable {var_input}")

        conf_stat = {
            "input_path": os.path.join(path_split_target_original_dir, f"{var_target}.train.txt"),
            "output_path": path_stat_target_original_dir,
            "number_of_threads": conf.multithread_operations.number_of_threads,
            "var_name": var_target,
        }

        conf_stat_path = os.path.join(path_conf_dir, f"conf.stat.{var_target}.json")
        with open(conf_stat_path, "w") as f:
            json.dump(conf_stat, f, indent=4)
        print(f"    Configuration file for stat created for target variable: {var_target}")
        print(f"    conf_file_path: {conf_stat_path}")

        log_file_path = os.path.join(path_log_dir, f"stat_{var_target}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", stat_script_path, "--config", conf_stat_path]
        execute_command(cmd, log_file_path, f"Stat for target variable {var_target}")


    list_conversion_variables = vars(conf.conversion_variables)
    print(list_conversion_variables)

    for var_input, var_target in list_conversion_variables.items():
        converted_var_input = f"{var_input}.{conf.conversion_type}"
        converted_var_target = f"{var_target}.{conf.conversion_type}"

        print(f"    converted_var_input: {converted_var_input}")
        print(f"    converted_var_target: {converted_var_target}")

        conf_stat = {
            "input_path": os.path.join(path_split_input_converted_dir, f"{converted_var_input}.train.txt"),
            "output_path": path_stat_input_converted_dir,
            "number_of_threads": conf.multithread_operations.number_of_threads,
            "var_name": var_input,
        }

        conf_stat_path = os.path.join(path_conf_dir, f"conf.stat.{converted_var_input}.json")
        with open(conf_stat_path, "w") as f:
            json.dump(conf_stat, f, indent=4)
        print(f"    Configuration file for stat created for converted input variable: {converted_var_input}")
        print(f"    conf_file_path: {conf_stat_path}")

        log_file_path = os.path.join(path_log_dir, f"stat_{converted_var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", stat_script_path, "--config", conf_stat_path]
        execute_command(cmd, log_file_path, f"Stat for converted input variable {converted_var_input}")

        conf_stat = {
            "input_path": os.path.join(path_split_target_converted_dir, f"{converted_var_target}.train.txt"),
            "output_path": path_stat_target_converted_dir,
            "number_of_threads": conf.multithread_operations.number_of_threads,
            "var_name": var_target,
        }

        conf_stat_path = os.path.join(path_conf_dir, f"conf.stat.{converted_var_target}.json")
        with open(conf_stat_path, "w") as f:
            json.dump(conf_stat, f, indent=4)
        print(f"    Configuration file for stat created for converted target variable: {converted_var_target}")
        print(f"    conf_file_path: {conf_stat_path}")

        log_file_path = os.path.join(path_log_dir, f"stat_{converted_var_target}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", stat_script_path, "--config", conf_stat_path]
        execute_command(cmd, log_file_path, f"Stat for converted target variable {converted_var_target}")


    list_treshold_variables = vars(conf.treshold_variables)
    print(list_treshold_variables)

    for var_input, var_target in list_treshold_variables.items():
        print(f"    var_input: {var_input}")
        print(f"    var_target: {var_target}")

        conf_stat = {
            "input_path": os.path.join(path_split_input_treshold_dir, f"{var_input}.train.txt"),
            "output_path": path_stat_input_treshold_dir,
            "number_of_threads": conf.multithread_operations.number_of_threads,
            "var_name": var_input,
        }

        conf_stat_path = os.path.join(path_conf_dir, f"conf.stat.treshold.{var_input}.json")
        with open(conf_stat_path, "w") as f:
            json.dump(conf_stat, f, indent=4)
        print(f"    Configuration file for stat created for tresholded input variable: {var_input}")
        print(f"    conf_file_path: {conf_stat_path}")

        log_file_path = os.path.join(path_log_dir, f"stat_treshold_{var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", stat_script_path, "--config", conf_stat_path]
        execute_command(cmd, log_file_path, f"Stat for tresholded input variable {var_input}")

        conf_stat = {
            "input_path": os.path.join(path_split_target_treshold_dir, f"{var_target}.train.txt"),
            "output_path": path_stat_target_treshold_dir,
            "number_of_threads": conf.multithread_operations.number_of_threads,
            "var_name": var_target,
        }

        conf_stat_path = os.path.join(path_conf_dir, f"conf.stat.treshold.{var_target}.json")
        with open(conf_stat_path, "w") as f:
            json.dump(conf_stat, f, indent=4)
        print(f"    Configuration file for stat created for tresholded target variable: {var_target}")
        print(f"    conf_file_path: {conf_stat_path}")

        log_file_path = os.path.join(path_log_dir, f"stat_treshold_{var_target}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", stat_script_path, "--config", conf_stat_path]
        execute_command(cmd, log_file_path, f"Stat for tresholded target variable {var_target}")


        converted_var_input = f"{var_input}.{conf.conversion_type}"
        converted_var_target = f"{var_target}.{conf.conversion_type}"

        conf_stat = {
            "input_path": os.path.join(path_split_input_treshold_converted_dir, f"{converted_var_input}.train.txt"),
            "output_path": path_stat_input_treshold_converted_dir,
            "number_of_threads": conf.multithread_operations.number_of_threads,
            "var_name": var_input,
        }

        conf_stat_path = os.path.join(path_conf_dir, f"conf.stat.treshold.{converted_var_input}.json")
        with open(conf_stat_path, "w") as f:
            json.dump(conf_stat, f, indent=4)
        print(f"    Configuration file for stat created for tresholded converted input variable: {converted_var_input}")
        print(f"    conf_file_path: {conf_stat_path}")

        log_file_path = os.path.join(path_log_dir, f"stat_treshold_{converted_var_input}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", stat_script_path, "--config", conf_stat_path]
        execute_command(cmd, log_file_path, f"Stat for tresholded converted input variable {converted_var_input}")

        conf_stat = {
            "input_path": os.path.join(path_split_target_treshold_converted_dir, f"{converted_var_target}.train.txt"),
            "output_path": path_stat_target_treshold_converted_dir,
            "number_of_threads": conf.multithread_operations.number_of_threads,
            "var_name": var_target,
        }

        conf_stat_path = os.path.join(path_conf_dir, f"conf.stat.treshold.{converted_var_target}.json")
        with open(conf_stat_path, "w") as f:
            json.dump(conf_stat, f, indent=4)
        print(f"    Configuration file for stat created for tresholded converted target variable: {converted_var_target}")
        print(f"    conf_file_path: {conf_stat_path}")

        log_file_path = os.path.join(path_log_dir, f"stat_treshold_{converted_var_target}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", stat_script_path, "--config", conf_stat_path]
        execute_command(cmd, log_file_path, f"Stat for tresholded converted target variable {converted_var_target}")
# end compute_statistics

def compute_pt(conf):
    split_label = conf.split.label

    path_conf_dir = os.path.join(conf.path_preproc_dir, "conf.files")
    make_pt_script_path = os.path.join(os.path.dirname(__file__), "make_pt.py")
    make_pt_rivers_script_path = os.path.join(os.path.dirname(__file__), "make_pt_rivers.py")
    path_log_dir = os.path.join(conf.path_preproc_dir, "log")
    path_pt_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "pt.files")

    path_split_input_interpolated_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "interpolated")
    path_split_input_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "converted")
    path_split_input_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "treshold")
    path_split_input_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "input", "treshold.converted")
    path_split_target_original_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "original")
    path_split_target_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "converted")
    path_split_target_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "treshold")
    path_split_target_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "target", "treshold.converted")
    path_split_rivers_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "rivers")

    path_stat_input_interpolated_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "interpolated")
    path_stat_input_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "converted")
    path_stat_input_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "treshold")
    path_stat_input_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.input", "treshold.converted")
    path_stat_target_original_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "original")
    path_stat_target_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "converted")
    path_stat_target_treshold_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "treshold")
    path_stat_target_treshold_converted_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "stat.target", "treshold.converted")

    # NOTE: statistics used for normalization are always taken from the train split,
    # regardless of which pt_type (train/val/test) is being produced.
    list_pt_types = ["train", "val", "test"]

    list_variables = vars(conf.variables)
    print(list_variables)

    for var_input, var_target in list_variables.items():
        for pt_type in list_pt_types:
            conf_pt = {
                "n_workers": conf.multithread_operations.number_of_threads,
                "path_target": os.path.join(path_split_target_original_dir, f"{var_target}.{pt_type}.txt"),
                "path_input": os.path.join(path_split_input_interpolated_dir, f"{var_input}.{pt_type}.txt"),
                "stat_target": os.path.join(path_stat_target_original_dir, f"stat.{var_target}.train.txt"),
                "stat_input": os.path.join(path_stat_input_interpolated_dir, f"stat.{var_input}.train.txt"),
                "var_target": var_target,
                "var_input": var_input,
                "output_path": path_pt_dir,
                "output_file_name": f"{var_input}.{var_target}.{pt_type}.dataset.pt",
            }

            output_file_path = os.path.join(conf_pt["output_path"], conf_pt["output_file_name"])
            if os.path.exists(output_file_path):
                print(f"Warning: output file {output_file_path} already exists. Skipping pt creation for variable pairing: {var_input} -> {var_target} ({pt_type})")
                continue

            conf_pt_path = os.path.join(path_conf_dir, f"conf.pt.{var_input}.{var_target}.{pt_type}.json")
            with open(conf_pt_path, "w") as f:
                json.dump(conf_pt, f, indent=4)
            print(f"    Configuration file for pt creation created for variable pairing: {var_input} -> {var_target} ({pt_type})")
            print(f"    conf_file_path: {conf_pt_path}")

            log_file_path = os.path.join(path_log_dir, f"pt_{var_input}_{var_target}_{pt_type}.log")
            print(f"    log_file_path: {log_file_path}")

            cmd = ["python", make_pt_script_path, "--config", conf_pt_path]
            # execute_command(cmd, log_file_path, f"Pt creation for variable pairing {var_input} -> {var_target} ({pt_type})")


    list_conversion_variables = vars(conf.conversion_variables)
    print(list_conversion_variables)

    for var_input, var_target in list_conversion_variables.items():
        converted_var_input = f"{var_input}.{conf.conversion_type}"
        converted_var_target = f"{var_target}.{conf.conversion_type}"

        for pt_type in list_pt_types:
            conf_pt = {
                "n_workers": conf.multithread_operations.number_of_threads,
                "path_target": os.path.join(path_split_target_converted_dir, f"{converted_var_target}.{pt_type}.txt"),
                "path_input": os.path.join(path_split_input_converted_dir, f"{converted_var_input}.{pt_type}.txt"),
                "stat_target": os.path.join(path_stat_target_converted_dir, f"stat.{converted_var_target}.train.txt"),
                "stat_input": os.path.join(path_stat_input_converted_dir, f"stat.{converted_var_input}.train.txt"),
                "var_target": var_target,
                "var_input": var_input,
                "output_path": path_pt_dir,
                "output_file_name": f"{converted_var_input}.{converted_var_target}.{pt_type}.dataset.pt",
            }

            output_file_path = os.path.join(conf_pt["output_path"], conf_pt["output_file_name"])
            if os.path.exists(output_file_path):
                print(f"Warning: output file {output_file_path} already exists. Skipping pt creation for converted variable pairing: {converted_var_input} -> {converted_var_target} ({pt_type})")
                continue

            conf_pt_path = os.path.join(path_conf_dir, f"conf.pt.{converted_var_input}.{converted_var_target}.{pt_type}.json")
            with open(conf_pt_path, "w") as f:
                json.dump(conf_pt, f, indent=4)
            print(f"    Configuration file for pt creation created for converted variable pairing: {converted_var_input} -> {converted_var_target} ({pt_type})")
            print(f"    conf_file_path: {conf_pt_path}")

            log_file_path = os.path.join(path_log_dir, f"pt_{converted_var_input}_{converted_var_target}_{pt_type}.log")
            print(f"    log_file_path: {log_file_path}")

            cmd = ["python", make_pt_script_path, "--config", conf_pt_path]
            # execute_command(cmd, log_file_path, f"Pt creation for converted variable pairing {converted_var_input} -> {converted_var_target} ({pt_type})")


    list_treshold_variables = vars(conf.treshold_variables)
    print(list_treshold_variables)

    for var_input, var_target in list_treshold_variables.items():
        for pt_type in list_pt_types:
            conf_pt = {
                "n_workers": conf.multithread_operations.number_of_threads,
                "path_target": os.path.join(path_split_target_treshold_dir, f"{var_target}.{pt_type}.txt"),
                "path_input": os.path.join(path_split_input_treshold_dir, f"{var_input}.{pt_type}.txt"),
                "stat_target": os.path.join(path_stat_target_treshold_dir, f"stat.{var_target}.train.txt"),
                "stat_input": os.path.join(path_stat_input_treshold_dir, f"stat.{var_input}.train.txt"),
                "var_target": var_target,
                "var_input": var_input,
                "output_path": path_pt_dir,
                "output_file_name": f"treshold.{var_input}.{var_target}.{pt_type}.dataset.pt",
            }

            output_file_path = os.path.join(conf_pt["output_path"], conf_pt["output_file_name"])
            if os.path.exists(output_file_path):
                print(f"Warning: output file {output_file_path} already exists. Skipping pt creation for tresholded variable pairing: {var_input} -> {var_target} ({pt_type})")
                continue

            conf_pt_path = os.path.join(path_conf_dir, f"conf.pt.treshold.{var_input}.{var_target}.{pt_type}.json")
            with open(conf_pt_path, "w") as f:
                json.dump(conf_pt, f, indent=4)
            print(f"    Configuration file for pt creation created for tresholded variable pairing: {var_input} -> {var_target} ({pt_type})")
            print(f"    conf_file_path: {conf_pt_path}")

            log_file_path = os.path.join(path_log_dir, f"pt_treshold_{var_input}_{var_target}_{pt_type}.log")
            print(f"    log_file_path: {log_file_path}")

            cmd = ["python", make_pt_script_path, "--config", conf_pt_path]
            # execute_command(cmd, log_file_path, f"Pt creation for tresholded variable pairing {var_input} -> {var_target} ({pt_type})")


        converted_var_input = f"{var_input}.{conf.conversion_type}"
        converted_var_target = f"{var_target}.{conf.conversion_type}"

        for pt_type in list_pt_types:
            conf_pt = {
                "n_workers": conf.multithread_operations.number_of_threads,
                "path_target": os.path.join(path_split_target_treshold_converted_dir, f"{converted_var_target}.{pt_type}.txt"),
                "path_input": os.path.join(path_split_input_treshold_converted_dir, f"{converted_var_input}.{pt_type}.txt"),
                "stat_target": os.path.join(path_stat_target_treshold_converted_dir, f"stat.{converted_var_target}.train.txt"),
                "stat_input": os.path.join(path_stat_input_treshold_converted_dir, f"stat.{converted_var_input}.train.txt"),
                "var_target": var_target,
                "var_input": var_input,
                "output_path": path_pt_dir,
                "output_file_name": f"treshold.{converted_var_input}.{converted_var_target}.{pt_type}.dataset.pt",
            }

            output_file_path = os.path.join(conf_pt["output_path"], conf_pt["output_file_name"])
            if os.path.exists(output_file_path):
                print(f"Warning: output file {output_file_path} already exists. Skipping pt creation for tresholded converted variable pairing: {converted_var_input} -> {converted_var_target} ({pt_type})")
                continue

            conf_pt_path = os.path.join(path_conf_dir, f"conf.pt.treshold.{converted_var_input}.{converted_var_target}.{pt_type}.json")
            with open(conf_pt_path, "w") as f:
                json.dump(conf_pt, f, indent=4)
            print(f"    Configuration file for pt creation created for tresholded converted variable pairing: {converted_var_input} -> {converted_var_target} ({pt_type})")
            print(f"    conf_file_path: {conf_pt_path}")

            log_file_path = os.path.join(path_log_dir, f"pt_treshold_{converted_var_input}_{converted_var_target}_{pt_type}.log")
            print(f"    log_file_path: {log_file_path}")

            cmd = ["python", make_pt_script_path, "--config", conf_pt_path]
            # execute_command(cmd, log_file_path, f"Pt creation for tresholded converted variable pairing {converted_var_input} -> {converted_var_target} ({pt_type})")


    for pt_type in list_pt_types:
        conf_pt_river = {
            "input_path": os.path.join(path_split_rivers_dir, f"rivers.{pt_type}.txt"),
            "output_path": path_pt_dir,
            "label": pt_type,
        }

        output_file_path = os.path.join(conf_pt_river["output_path"], f"rivers_{pt_type}.pt")
        if os.path.exists(output_file_path):
            print(f"Warning: output file {output_file_path} already exists. Skipping pt creation for rivers ({pt_type})")
            continue

        conf_pt_river_path = os.path.join(path_conf_dir, f"conf.pt.rivers.{pt_type}.json")
        with open(conf_pt_river_path, "w") as f:
            json.dump(conf_pt_river, f, indent=4)
        print(f"    Configuration file for pt creation created for rivers ({pt_type})")
        print(f"    conf_file_path: {conf_pt_river_path}")

        log_file_path = os.path.join(path_log_dir, f"pt_rivers_{pt_type}.log")
        print(f"    log_file_path: {log_file_path}")

        cmd = ["python", make_pt_rivers_script_path, "--config", conf_pt_river_path]
        execute_command(cmd, log_file_path, f"Pt creation for rivers ({pt_type})")
# end compute_pt


if __name__ == "__main__":
    conf = read_conf_file(path_conf_file)

    print(f"path_target_dir: {conf.path_target_dir}")
    print(f"path_input_dir: {conf.path_input_dir}")
    print(f"path_preproc_dir: {conf.path_preproc_dir}")
    print(f"conversion_type: {conf.conversion_type}")
    print(f"number_of_threads: {conf.multithread_operations.number_of_threads}")
    print(f"variables: {conf.variables}")
    print(f"conversion_variables: {conf.conversion_variables}")
    print(f"split: {conf.split}")
    print(f"split_flag: {conf.multithread_operations.split_flag}")
    print(f"compute_statistics_flag: {conf.multithread_operations.compute_statistics_flag}")
    print(f"compute_pt_flag: {conf.multithread_operations.compute_pt_flag}")

    if conf.multithread_operations.split_flag == True:
        run_split_layer(conf)

    if conf.multithread_operations.compute_statistics_flag == True:
        compute_statistics(conf)

    if conf.multithread_operations.compute_pt_flag == True:
        compute_pt(conf)
