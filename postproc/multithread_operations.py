import argparse
import glob
import json
import os
import shlex

from pp_library import read_file_list
from pp_library_conf import read_multithread_operations_conf


def resolve_postproc_path(relative_path):
    return os.path.join(os.path.dirname(__file__), relative_path)


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Write test-set file lists and compute monthly/seasonal analytics."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf.files.dir", "conf_orchestrator.json"),
        help="Path to the postproc orchestrator configuration file.",
    )
    return parser.parse_args()


def read_data_type_specs(path):
    with open(path) as f:
        tree = json.load(f)["tree"]
    return [
        (category, spec["variables_key"], spec["suffixed"], spec["pt_prefix"])
        for category, spec in tree.items()
    ]


def build_pair_name(var_input, var_target, conversion_type, suffixed):
    if suffixed:
        return f"{var_input}.{conversion_type}.{var_target}.{conversion_type}"
    return f"{var_input}.{var_target}"


def input_subdir_name(category):
    return "interpolated" if category == "original" else category


def analytics_category_name(group, category):
    if group == "input" and category == "original":
        return "interpolated"
    return category


def list_source_files(conf, split_label, group, category, input_subdir, names_input, names_target, var_target, pair_name, stage=None):
    if group == "input":
        list_path = os.path.join(
            conf.path_preproc_dir, "splits", split_label, "input", input_subdir, f"{names_input}.test.txt"
        )
        return read_file_list(list_path)

    if group == "target":
        list_path = os.path.join(
            conf.path_preproc_dir, "splits", split_label, "target", category, f"{names_target}.test.txt"
        )
        return read_file_list(list_path)

    predictions_dir = os.path.join(
        conf.path_postproc_dir, "predictions", "test.dataset", category,
        *filter(None, [stage]), pair_name, var_target
    )
    return sorted(glob.glob(os.path.join(predictions_dir, "*.nc")))


def write_file_list(list_path, files):
    os.makedirs(os.path.dirname(list_path), exist_ok=True)
    with open(list_path, "w") as f:
        for file_path in files:
            f.write(f"{file_path}\n")


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


def process_pair_analytics(conf, mt_conf, paths, split_label, jobs, group, category, stage,
                            input_subdir, var_input, names_input, names_target, var_target, pair_name, analytics_category):
    (split_months_path, split_seasons_path, compute_monthly_stats_path,
     compute_season_stat_field_path, path_conf_dir, path_log_dir) = paths
    statistics = vars(mt_conf.statistics)

    variable_name = var_input if group == "input" else var_target
    label_base = f"{group}.{category}.{pair_name}" if stage is None else f"{group}.{category}.{stage}.{pair_name}"

    files = list_source_files(
        conf, split_label, group, category, input_subdir, names_input, names_target, var_target, pair_name, stage
    )
    if not files:
        print(f"Warning: no source files found, skipping: {label_base}")
        return

    file_list_dir = os.path.join(
        conf.path_postproc_dir, "file.list", "test.dataset", group, analytics_category, *filter(None, [stage])
    )
    analytics_dir = os.path.join(
        conf.path_postproc_dir, "analytics", "test.dataset", group, analytics_category, *filter(None, [stage])
    )

    flat_list_path = os.path.join(file_list_dir, f"{pair_name}.txt")
    write_file_list(flat_list_path, files)

    if statistics.get("monthly", False):
        # split into months
        conf_split_months = {
            "input_path": flat_list_path,
            "output_path": file_list_dir,
            "output_folder_name": os.path.join("montly", pair_name),
        }
        conf_path = os.path.join(path_conf_dir, f"conf.split_months.{label_base}.json")
        write_conf(conf_path, conf_split_months)
        log_file_path = os.path.join(path_log_dir, f"split_months_{label_base}.log")
        execute_command(["python", split_months_path, "--config", conf_path], log_file_path, f"split_months for {label_base}")

        # monthly stats
        conf_compute_monthly_stats = {
            "input_path": os.path.join(file_list_dir, "montly", pair_name),
            "output_path": analytics_dir,
            "output_folder_name": os.path.join("montly", pair_name),
            "variable": variable_name,
            "jobs": jobs,
        }
        conf_path = os.path.join(path_conf_dir, f"conf.compute_monthly_stats.{label_base}.json")
        write_conf(conf_path, conf_compute_monthly_stats)
        log_file_path = os.path.join(path_log_dir, f"compute_monthly_stats_{label_base}.log")
        execute_command(["python", compute_monthly_stats_path, "--config", conf_path], log_file_path, f"compute_monthly_stats for {label_base}")

    if statistics.get("seasonal", False):
        # split into seasons
        conf_split_seasons = {
            "input_path": flat_list_path,
            "output_path": file_list_dir,
            "output_folder_name": os.path.join("seasonal", pair_name),
        }
        conf_path = os.path.join(path_conf_dir, f"conf.split_seasons.{label_base}.json")
        write_conf(conf_path, conf_split_seasons)
        log_file_path = os.path.join(path_log_dir, f"split_seasons_{label_base}.log")
        execute_command(["python", split_seasons_path, "--config", conf_path], log_file_path, f"split_seasons for {label_base}")

        # seasonal stat fields
        conf_compute_season_stat_field = {
            "input_path": os.path.join(file_list_dir, "seasonal", pair_name),
            "output_path": analytics_dir,
            "output_folder_name": os.path.join("seasonal", pair_name),
            "variable": variable_name,
            "jobs": jobs,
        }
        conf_path = os.path.join(path_conf_dir, f"conf.compute_season_stat_field.{label_base}.json")
        write_conf(conf_path, conf_compute_season_stat_field)
        log_file_path = os.path.join(path_log_dir, f"compute_season_stat_field_{label_base}.log")
        execute_command(["python", compute_season_stat_field_path, "--config", conf_path], log_file_path, f"compute_season_stat_field for {label_base}")


def run_analytics(conf):
    mt_conf = conf.multithread_operations
    split_label = conf.split_label
    jobs = mt_conf.number_of_jobs
    groups = vars(mt_conf.groups)
    scripts = vars(mt_conf.scripts)
    staged_categories = vars(conf.mpi_operations.categories)

    training_structure_path = resolve_postproc_path(mt_conf.training_structure_path)
    path_conf_dir = os.path.join(conf.path_postproc_dir, "conf.files")
    path_log_dir = os.path.join(conf.path_postproc_dir, "log")
    os.makedirs(path_conf_dir, exist_ok=True)
    os.makedirs(path_log_dir, exist_ok=True)

    paths = (
        resolve_postproc_path(scripts["split_months"]),
        resolve_postproc_path(scripts["split_seasons"]),
        resolve_postproc_path(scripts["compute_monthly_stats"]),
        resolve_postproc_path(scripts["compute_season_stat_field"]),
        path_conf_dir,
        path_log_dir,
    )

    for category, variables_key, suffixed, _ in read_data_type_specs(training_structure_path):
        variables = vars(getattr(conf, variables_key))
        input_subdir = input_subdir_name(category)

        for var_input, var_target in variables.items():
            names_input = f"{var_input}.{conf.conversion_type}" if suffixed else var_input
            names_target = f"{var_target}.{conf.conversion_type}" if suffixed else var_target
            pair_name = build_pair_name(var_input, var_target, conf.conversion_type, suffixed)

            for group, group_enabled in groups.items():
                if not group_enabled:
                    continue

                analytics_category = analytics_category_name(group, category)

                if group == "predictions" and category in staged_categories:
                    conversion_type = vars(staged_categories[category])["conversion_type"]
                    stages = ["raw", f"treshold.{conversion_type}"]
                else:
                    stages = [None]

                for stage in stages:
                    process_pair_analytics(
                        conf, mt_conf, paths, split_label, jobs, group, category, stage,
                        input_subdir, var_input, names_input, names_target, var_target, pair_name, analytics_category
                    )


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_multithread_operations_conf(args.config)

    run_analytics(conf)
