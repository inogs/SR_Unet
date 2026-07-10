import json
import math
import os
import re
import subprocess
from pathlib import Path


path_workdir = "/leonardo_scratch/large/userexternal/gzuccari"
path_opa_home = os.path.join(path_workdir, "OPA_HOME")
path_test2 = os.path.join(path_workdir, "TEST_2")
path_test2_run = os.path.join(path_test2, "06.30.test.dir.MID")
path_predictions = os.path.join(path_test2, "predictions")
path_predictions_v3 = os.path.join(path_test2, "predictions.v3")
path_postprocessing = os.path.join(path_test2, "postprocessing")

number_of_processes = 48

produce_predictions_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "produce_predictions_prod.py",
)

converter_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "pipeline",
    "converter_mpi.py",
)

apply_treshold_path = os.path.join(
    os.path.dirname(__file__),
    "apply_treshold_mpi.py",
)

thresholds = {
    "Chla": 0.00001,
    "N1p": 0.001,
    "N3n": 0.001,
}

prediction_entries = [
    {
        "name": "chl.Chla",
        "input_variable": "chl",
        "target_variable": "Chla",
        "test_path": os.path.join(path_opa_home, "pt.files", "chl.Chla.test.dataset.pt"),
        "reference_nc_list": os.path.join(path_opa_home, "input", "split", "original.variables", "chl.test.txt"),
        "reference_target_nc_list": os.path.join(path_opa_home, "target", "split", "original.variables", "Chla.test.txt"),
        "stats_file": os.path.join(path_opa_home, "target", "stat", "original.variables", "stat.Chla.train.txt"),
    },
    {
        "name": "chl.log.Chla.log",
        "input_variable": "chl",
        "target_variable": "Chla",
        "test_path": os.path.join(path_opa_home, "pt.files", "chl.log.Chla.log.test.pt"),
        "reference_nc_list": os.path.join(path_opa_home, "input", "split", "converted.variables", "chl.log.test.txt"),
        "reference_target_nc_list": os.path.join(path_opa_home, "target", "split", "converted.variables", "Chla.log.test.txt"),
        "stats_file": os.path.join(path_opa_home, "target", "stat", "converted.variables", "stat.Chla.log.train.txt"),
    },
    {
        "name": "no3.N3n",
        "input_variable": "no3",
        "target_variable": "N3n",
        "test_path": os.path.join(path_opa_home, "pt.files", "no3.N3n.test.dataset.pt"),
        "reference_nc_list": os.path.join(path_opa_home, "input", "split", "original.variables", "no3.test.txt"),
        "reference_target_nc_list": os.path.join(path_opa_home, "target", "split", "original.variables", "N3n.test.txt"),
        "stats_file": os.path.join(path_opa_home, "target", "stat", "original.variables", "stat.N3n.train.txt"),
    },
    {
        "name": "no3.log.N3n.log",
        "input_variable": "no3",
        "target_variable": "N3n",
        "test_path": os.path.join(path_opa_home, "pt.files", "no3.log.N3n.log.test.pt"),
        "reference_nc_list": os.path.join(path_opa_home, "input", "split", "converted.variables", "no3.log.test.txt"),
        "reference_target_nc_list": os.path.join(path_opa_home, "target", "split", "converted.variables", "N3n.log.test.txt"),
        "stats_file": os.path.join(path_opa_home, "target", "stat", "converted.variables", "stat.N3n.log.train.txt"),
    },
    {
        "name": "po4.N1p",
        "input_variable": "po4",
        "target_variable": "N1p",
        "test_path": os.path.join(path_opa_home, "pt.files", "po4.N1p.test.dataset.pt"),
        "reference_nc_list": os.path.join(path_opa_home, "input", "split", "original.variables", "po4.test.txt"),
        "reference_target_nc_list": os.path.join(path_opa_home, "target", "split", "original.variables", "N1p.test.txt"),
        "stats_file": os.path.join(path_opa_home, "target", "stat", "original.variables", "stat.N1p.train.txt"),
    },
    {
        "name": "po4.log.N1p.log",
        "input_variable": "po4",
        "target_variable": "N1p",
        "test_path": os.path.join(path_opa_home, "pt.files", "po4.log.N1p.log.test.pt"),
        "reference_nc_list": os.path.join(path_opa_home, "input", "split", "converted.variables", "po4.log.test.txt"),
        "reference_target_nc_list": os.path.join(path_opa_home, "target", "split", "converted.variables", "N1p.log.test.txt"),
        "stats_file": os.path.join(path_opa_home, "target", "stat", "converted.variables", "stat.N1p.log.train.txt"),
    },
    {
        "name": "so.S",
        "input_variable": "so",
        "target_variable": "S",
        "test_path": os.path.join(path_opa_home, "pt.files", "so.S.test.dataset.pt"),
        "reference_nc_list": os.path.join(path_opa_home, "input", "split", "original.variables", "so.test.txt"),
        "reference_target_nc_list": os.path.join(path_opa_home, "target", "split", "original.variables", "S.test.txt"),
        "stats_file": os.path.join(path_opa_home, "target", "stat", "original.variables", "stat.S.train.txt"),
    },
    {
        "name": "thetao.T",
        "input_variable": "thetao",
        "target_variable": "T",
        "test_path": os.path.join(path_opa_home, "pt.files", "thetao.T.test.dataset.pt"),
        "reference_nc_list": os.path.join(path_opa_home, "input", "split", "original.variables", "thetao.test.txt"),
        "reference_target_nc_list": os.path.join(path_opa_home, "target", "split", "original.variables", "T.test.txt"),
        "stats_file": os.path.join(path_opa_home, "target", "stat", "original.variables", "stat.T.train.txt"),
    },
]

target_entries = [
    {
        "name": "Chla",
        "variable_name": "Chla",
        "threshold_key": "Chla",
        "source_list": os.path.join(path_opa_home, "target", "split", "original.variables", "Chla.test.txt"),
    },
    {
        "name": "N1p",
        "variable_name": "N1p",
        "threshold_key": "N1p",
        "source_list": os.path.join(path_opa_home, "target", "split", "original.variables", "N1p.test.txt"),
    },
    {
        "name": "N3n",
        "variable_name": "N3n",
        "threshold_key": "N3n",
        "source_list": os.path.join(path_opa_home, "target", "split", "original.variables", "N3n.test.txt"),
    },
]

input_entries = [
    {
        "name": "chl",
        "variable_name": "chl",
        "threshold_key": "Chla",
        "source_list": os.path.join(path_opa_home, "input", "split", "original.variables", "chl.test.txt"),
    },
    {
        "name": "po4",
        "variable_name": "po4",
        "threshold_key": "N1p",
        "source_list": os.path.join(path_opa_home, "input", "split", "original.variables", "po4.test.txt"),
    },
    {
        "name": "no3",
        "variable_name": "no3",
        "threshold_key": "N3n",
        "source_list": os.path.join(path_opa_home, "input", "split", "original.variables", "no3.test.txt"),
    },
]

normal_net_entries = [
    {
        "name": "chl.Chla",
        "variable_name": "Chla",
        "threshold_key": "Chla",
        "source_folder": os.path.join(path_predictions, "chl.Chla", "Chla"),
    },
    {
        "name": "po4.N1p",
        "variable_name": "N1p",
        "threshold_key": "N1p",
        "source_folder": os.path.join(path_predictions, "po4.N1p", "N1p"),
    },
    {
        "name": "no3.N3n",
        "variable_name": "N3n",
        "threshold_key": "N3n",
        "source_folder": os.path.join(path_predictions, "no3.N3n", "N3n"),
    },
]

log_net_entries = [
    {
        "name": "chl.log.Chla.log",
        "variable_name": "Chla",
        "threshold_key": "Chla",
        "source_folder": os.path.join(path_predictions, "chl.log.Chla.log", "Chla"),
    },
    {
        "name": "po4.log.N1p.log",
        "variable_name": "N1p",
        "threshold_key": "N1p",
        "source_folder": os.path.join(path_predictions, "po4.log.N1p.log", "N1p"),
    },
    {
        "name": "no3.log.N3n.log",
        "variable_name": "N3n",
        "threshold_key": "N3n",
        "source_folder": os.path.join(path_predictions, "no3.log.N3n.log", "N3n"),
    },
]


def run_mpi(script_path, conf_path):
    cmd = [
        "mpirun",
        "-np",
        str(number_of_processes),
        "python",
        script_path,
        "--config",
        conf_path,
    ]
    subprocess.run(cmd, check=True)


def run_prediction(conf_path):
    subprocess.run(
        ["python", produce_predictions_path, "-c", conf_path],
        check=True,
    )


def read_file_list(file_list_path):
    with open(file_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def write_json(path, conf):
    with open(path, "w") as f:
        json.dump(conf, f, indent=4)
        f.write("\n")


def get_epoch_end(checkpoint_name):
    match = re.search(r"_(\d+)\.(\d+)\.ckpt$", checkpoint_name)
    if not match:
        return -1
    return int(match.group(2))


def get_max_epoch_weight(entry_name):
    out_dir = os.path.join(path_test2_run, entry_name, "out1")
    candidates = [
        os.path.join(out_dir, file_name)
        for file_name in os.listdir(out_dir)
        if file_name.startswith(f"best_model_{entry_name}_") and file_name.endswith(".ckpt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No epoch backup checkpoint found for {entry_name}: {out_dir}")
    return max(candidates, key=lambda item: get_epoch_end(os.path.basename(item)))


def make_prediction_conf(entry):
    return {
        "weight_file": get_max_epoch_weight(entry["name"]),
        "test_path": entry["test_path"],
        "output_path": path_predictions,
        "prediction_subdir": entry["name"],
        "variables": [entry["input_variable"]],
        "reference_nc_lists": {
            entry["input_variable"]: entry["reference_nc_list"],
        },
        "target_variables": {
            entry["input_variable"]: entry["target_variable"],
        },
        "reference_target_nc_lists": {
            entry["target_variable"]: entry["reference_target_nc_list"],
        },
        "stats_files": {
            entry["input_variable"]: entry["stats_file"],
        },
        "river_path": None,
        "batch_size": 4,
        "mask_threshold": 1000000.0,
        "device": "auto",
        "seed": 0,
    }


def make_predictions():
    os.makedirs(path_predictions, exist_ok=True)
    for entry in prediction_entries:
        conf = make_prediction_conf(entry)
        conf_path = os.path.join(path_predictions, f"{entry['name']}_postproc.json")
        write_json(conf_path, conf)
        print(f"Prediction configuration written: {conf_path}")
        run_prediction(conf_path)


def make_link(source_path, link_path):
    source_path = os.path.abspath(source_path)
    if os.path.lexists(link_path):
        if os.path.islink(link_path) and os.path.abspath(os.readlink(link_path)) == source_path:
            return
        raise FileExistsError(f"Path already exists and is not the expected link: {link_path}")
    os.symlink(source_path, link_path)


def link_from_file_list(file_list_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for source_file in read_file_list(file_list_path):
        make_link(source_file, os.path.join(output_dir, os.path.basename(source_file)))


def link_from_folder(source_dir, output_dir):
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source folder not found: {source_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for file_name in sorted(os.listdir(source_dir)):
        if file_name.endswith(".nc"):
            make_link(
                os.path.join(source_dir, file_name),
                os.path.join(output_dir, file_name),
            )


def write_file_list(folder_path, file_list_path):
    os.makedirs(os.path.dirname(file_list_path), exist_ok=True)
    nc_files = sorted(
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.endswith(".nc")
    )
    with open(file_list_path, "w") as f:
        for nc_file in nc_files:
            f.write(f"{nc_file}\n")


def write_conf(conf_path, conf):
    write_json(conf_path, conf)
    print("Configuration file created:")
    print(f"    {conf_path}")


def remove_conf(conf_path):
    os.remove(conf_path)
    print("Configuration file removed:")
    print(f"    {conf_path}")


def apply_treshold(folder_path, output_path, variable_name, treshold, conf_path):
    os.makedirs(output_path, exist_ok=True)
    conf = {
        "folder_path": folder_path,
        "output_path": output_path,
        "variable_name": variable_name,
        "treshold": treshold,
    }
    write_conf(conf_path, conf)
    run_mpi(apply_treshold_path, conf_path)
    remove_conf(conf_path)


def apply_log(folder_path, output_path, variable_name, conf_path):
    os.makedirs(output_path, exist_ok=True)
    conf = {
        "folder_path": folder_path,
        "output_path": output_path,
        "conversion_type": "log",
        "variable_name": variable_name,
    }
    write_conf(conf_path, conf)
    run_mpi(converter_path, conf_path)
    remove_conf(conf_path)


def link_already_log_treshold(folder_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".treshold.nc"):
            continue
        link_name = file_name.replace(".treshold.nc", ".treshold.log.nc")
        make_link(
            os.path.join(folder_path, file_name),
            os.path.join(output_path, link_name),
        )


def process_domain(domain_name, entries, source_kind, already_log=False):
    domain_path = os.path.join(path_predictions_v3, domain_name)
    original_root = os.path.join(domain_path, "original")
    treshold_root = os.path.join(domain_path, "treshold")
    treshold_log_root = os.path.join(domain_path, "treshold.log")
    file_lists_root = os.path.join(domain_path, "file.lists")

    for entry in entries:
        name = entry["name"]
        original_path = os.path.join(original_root, name)
        treshold_path = os.path.join(treshold_root, name)
        treshold_log_path = os.path.join(treshold_log_root, name)

        if source_kind == "file_list":
            link_from_file_list(entry["source_list"], original_path)
        elif source_kind == "folder":
            link_from_folder(entry["source_folder"], original_path)
        else:
            raise ValueError(f"Unknown source kind: {source_kind}")

        write_file_list(
            original_path,
            os.path.join(file_lists_root, "original", f"{name}.txt"),
        )

        treshold_value = thresholds[entry["threshold_key"]]
        if already_log:
            treshold_value = math.log(treshold_value)

        conf_treshold_path = os.path.join(
            treshold_root, f"conf.treshold.{domain_name}.{name}.json"
        )
        apply_treshold(
            folder_path=original_path,
            output_path=treshold_path,
            variable_name=entry["variable_name"],
            treshold=treshold_value,
            conf_path=conf_treshold_path,
        )
        write_file_list(
            treshold_path,
            os.path.join(file_lists_root, "treshold", f"{name}.txt"),
        )

        if already_log:
            link_already_log_treshold(treshold_path, treshold_log_path)
        else:
            conf_log_path = os.path.join(
                treshold_log_root, f"conf.treshold.log.{domain_name}.{name}.json"
            )
            apply_log(
                folder_path=treshold_path,
                output_path=treshold_log_path,
                variable_name=entry["variable_name"],
                conf_path=conf_log_path,
            )

        write_file_list(
            treshold_log_path,
            os.path.join(file_lists_root, "treshold.log", f"{name}.txt"),
        )


def process_predictions_v3():
    Path(path_predictions_v3).mkdir(parents=True, exist_ok=True)
    process_domain("target", target_entries, source_kind="file_list")
    process_domain("input", input_entries, source_kind="file_list")
    process_domain("normal.net", normal_net_entries, source_kind="folder")
    process_domain("log.net", log_net_entries, source_kind="folder", already_log=True)


if __name__ == "__main__":
    if not os.path.isdir(path_test2_run):
        raise FileNotFoundError(f"TEST_2 run folder not found: {path_test2_run}")
    if not os.path.isdir(path_opa_home):
        raise FileNotFoundError(f"OPA_HOME folder not found: {path_opa_home}")

    Path(path_postprocessing).mkdir(parents=True, exist_ok=True)
    # make_predictions()
    # process_predictions_v3()
