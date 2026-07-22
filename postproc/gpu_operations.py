import argparse
import glob
import json
import os
import shlex

from pp_library_conf import read_gpu_operations_conf


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
        description="Produce all test-set predictions on a single GPU, one network after another."
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


def input_subdir_name(category):
    return "interpolated" if category == "original" else category


def find_weight_file(out_dir):
    candidates = sorted(glob.glob(os.path.join(out_dir, "best_*.ckpt")))
    return candidates[0] if candidates else None


def execute_command(cmd, log_file_path, label):
    command = " ".join(shlex.quote(str(part)) for part in cmd)
    command_with_log = f"{command} > {shlex.quote(log_file_path)} 2>&1"

    print(f"    Running command: {command}")
    return_code = os.system(command_with_log)

    if return_code == 0:
        print(f"    {label} completed successfully.")
        return True

    exit_code = os.waitstatus_to_exitcode(return_code)
    print(f"Warning: {label} failed with return code {exit_code}. See log file: {log_file_path}")
    return False


def run_predictions(conf):
    gpu_conf = conf.gpu_operations
    if gpu_conf.n_devices != 1:
        raise ValueError(
            f"gpu_operations runs predictions serially on a single device, got n_devices={gpu_conf.n_devices}"
        )

    training_structure_path = resolve_postproc_path(gpu_conf.training_structure_path)
    produce_predictions_path = resolve_postproc_path(gpu_conf.produce_predictions_path)

    split_label = conf.split_label
    path_pt_dir = os.path.join(conf.path_preproc_dir, "splits", split_label, "pt.files")
    path_conf_dir = os.path.join(conf.path_postproc_dir, "conf.files")
    path_log_dir = os.path.join(conf.path_postproc_dir, "log")

    rivers_test_path = os.path.join(path_pt_dir, "rivers_test.pt")
    river_path = rivers_test_path if os.path.isfile(rivers_test_path) else None

    for category, variables_key, suffixed, pt_prefix in read_data_type_specs(training_structure_path):
        variables = vars(getattr(conf, variables_key))
        input_subdir = input_subdir_name(category)

        for var_input, var_target in variables.items():
            names_input = f"{var_input}.{conf.conversion_type}" if suffixed else var_input
            names_target = f"{var_target}.{conf.conversion_type}" if suffixed else var_target

            pair_name = build_pair_name(var_input, var_target, conf.conversion_type, suffixed)
            pt_basename = f"{pt_prefix}{pair_name}"
            label = f"Prediction for {category}/{pair_name}"

            out_dir = os.path.join(conf.path_training_dir, category, pair_name, "out")
            weight_file = find_weight_file(out_dir)
            if weight_file is None:
                print(f"Warning: no checkpoint found, skipping {label}: {out_dir}")
                continue

            prediction_conf = {
                "weight_file": weight_file,
                "test_path": os.path.join(path_pt_dir, f"{pt_basename}.test.dataset.pt"),
                "output_path": os.path.join(
                    conf.path_postproc_dir, "predictions", "test.dataset", category, pair_name
                ),
                "variables": [var_input],
                "reference_nc_lists": {
                    var_input: os.path.join(
                        conf.path_preproc_dir, "splits", split_label, "input", input_subdir, f"{names_input}.test.txt"
                    ),
                },
                "target_variables": {var_input: var_target},
                "reference_target_nc_lists": {
                    var_target: os.path.join(
                        conf.path_preproc_dir, "splits", split_label, "target", category, f"{names_target}.test.txt"
                    ),
                },
                "stats_files": {
                    var_input: os.path.join(
                        conf.path_preproc_dir,
                        "splits",
                        split_label,
                        "stat.target",
                        category,
                        f"stat.{names_target}.train.json",
                    ),
                },
                "river_path": river_path,
                "batch_size": gpu_conf.batch_size,
                "mask_threshold": gpu_conf.mask_threshold,
                "device": gpu_conf.device,
                "seed": gpu_conf.seed,
            }

            conf_path = os.path.join(path_conf_dir, f"conf.predict.{category}.{pair_name}.json")
            with open(conf_path, "w") as f:
                json.dump(prediction_conf, f, indent=4)
                f.write("\n")

            log_file_path = os.path.join(path_log_dir, f"predict_{category}_{pair_name}.log")

            cmd = ["python", produce_predictions_path, "-c", conf_path]
            execute_command(cmd, log_file_path, label)


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_gpu_operations_conf(args.config)

    run_predictions(conf)
