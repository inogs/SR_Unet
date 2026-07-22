import argparse
import json
import math
import os
import shlex

from pp_library_conf import read_mpi_operations_conf


def resolve_postproc_path(relative_path):
    return os.path.join(os.path.dirname(__file__), relative_path)


def read_data_type_specs(path):
    with open(path) as f:
        tree = json.load(f)["tree"]
    return [
        (category, spec["variables_key"], spec["suffixed"], spec["pt_prefix"])
        for category, spec in tree.items()
    ]


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Apply treshold and log/exp conversion to raw network predictions."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf.files.dir", "conf_orchestrator.json"),
        help="Path to the postproc orchestrator configuration file.",
    )
    return parser.parse_args()


def build_pair_name(var_input, var_target, conversion_type, suffixed):
    if suffixed:
        return f"{var_input}.{conversion_type}.{var_target}.{conversion_type}"
    return f"{var_input}.{var_target}"


def write_conf(conf_path, conf_dict):
    os.makedirs(os.path.dirname(conf_path), exist_ok=True)
    with open(conf_path, "w") as f:
        json.dump(conf_dict, f, indent=4)
        f.write("\n")


def execute_command(cmd, log_file_path, label):
    command = " ".join(shlex.quote(str(part)) for part in cmd)
    command_with_log = f"{command} > {shlex.quote(log_file_path)} 2>&1"

    print(f"    Running {label}: {command}")
    return_code = os.system(command_with_log)

    if return_code == 0:
        print(f"    {label} completed successfully.")
        return True

    exit_code = os.waitstatus_to_exitcode(return_code)
    print(f"Warning: {label} failed with return code {exit_code}. See log file: {log_file_path}")
    return False


def run_mpi_operations(conf):
    mpi_conf = conf.mpi_operations
    training_structure_path = resolve_postproc_path(mpi_conf.training_structure_path)
    apply_treshold_path = resolve_postproc_path(mpi_conf.apply_treshold_path)
    converter_path = resolve_postproc_path(mpi_conf.converter_path)
    number_of_processes = mpi_conf.number_of_processes
    thresholds = vars(mpi_conf.thresholds)
    categories = vars(mpi_conf.categories)

    path_conf_dir = os.path.join(conf.path_postproc_dir, "conf.files")
    path_log_dir = os.path.join(conf.path_postproc_dir, "log")
    os.makedirs(path_conf_dir, exist_ok=True)
    os.makedirs(path_log_dir, exist_ok=True)

    for category, variables_key, suffixed, _ in read_data_type_specs(training_structure_path):
        if category not in categories:
            continue

        stage_conf = vars(categories[category])
        conversion_type = stage_conf["conversion_type"]
        threshold_in_log_space = stage_conf["threshold_in_log_space"]
        final_stage = f"treshold.{conversion_type}"

        variables = vars(getattr(conf, variables_key))

        for var_input, var_target in variables.items():
            threshold = thresholds.get(var_target)
            if threshold is None:
                print(f"Warning: no threshold defined for {var_target}, skipping {category}/{var_input}.{var_target}")
                continue
            if threshold_in_log_space:
                threshold = math.log(threshold)

            pair_name = build_pair_name(var_input, var_target, conf.conversion_type, suffixed)
            label_base = f"{category}.{pair_name}"

            raw_dir = os.path.join(
                conf.path_postproc_dir, "predictions", "test.dataset", category, "raw", pair_name, var_target
            )
            if not os.path.isdir(raw_dir):
                print(f"Warning: no raw predictions found, skipping: {label_base}: {raw_dir}")
                continue

            treshold_dir = os.path.join(
                conf.path_postproc_dir, "predictions", "test.dataset", category, "treshold", pair_name, var_target
            )
            final_dir = os.path.join(
                conf.path_postproc_dir, "predictions", "test.dataset", category, final_stage, pair_name, var_target
            )

            # step 1: treshold
            conf_treshold = {
                "folder_path": raw_dir,
                "output_path": treshold_dir,
                "variable_name": var_target,
                "treshold": threshold,
            }
            conf_path = os.path.join(path_conf_dir, f"conf.treshold.{label_base}.json")
            write_conf(conf_path, conf_treshold)
            log_file_path = os.path.join(path_log_dir, f"treshold_{label_base}.log")
            cmd = ["mpirun", "-np", str(number_of_processes), "python", apply_treshold_path, "--config", conf_path]
            execute_command(cmd, log_file_path, f"treshold for {label_base}")

            # step 2: conversion
            conf_conversion = {
                "folder_path": treshold_dir,
                "output_path": final_dir,
                "conversion_type": conversion_type,
                "variable_name": var_target,
            }
            conf_path = os.path.join(path_conf_dir, f"conf.{conversion_type}.{label_base}.json")
            write_conf(conf_path, conf_conversion)
            log_file_path = os.path.join(path_log_dir, f"{conversion_type}_{label_base}.log")
            cmd = ["mpirun", "-np", str(number_of_processes), "python", converter_path, "--config", conf_path]
            execute_command(cmd, log_file_path, f"{conversion_type} for {label_base}")


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_mpi_operations_conf(args.config)

    run_mpi_operations(conf)
