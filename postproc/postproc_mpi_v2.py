import json
import math
import os
import subprocess
from pathlib import Path


path_predictions = "/leonardo_scratch/large/userexternal/gzuccari/TEST/predictions"
path_predictions_v2 = "/leonardo_scratch/large/userexternal/gzuccari/TEST/predictions.v2"
path_opa_home = "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME"

number_of_processes = 32

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

target_entries = [
    {
        "name": "Chla",
        "variable_name": "Chla",
        "threshold_key": "Chla",
        "source_list": os.path.join(
            path_opa_home, "target", "split", "original.variables", "Chla.test.txt"
        ),
    },
    {
        "name": "N1p",
        "variable_name": "N1p",
        "threshold_key": "N1p",
        "source_list": os.path.join(
            path_opa_home, "target", "split", "original.variables", "N1p.test.txt"
        ),
    },
    {
        "name": "N3n",
        "variable_name": "N3n",
        "threshold_key": "N3n",
        "source_list": os.path.join(
            path_opa_home, "target", "split", "original.variables", "N3n.test.txt"
        ),
    },
]

input_entries = [
    {
        "name": "chl",
        "variable_name": "chl",
        "threshold_key": "Chla",
        "source_list": os.path.join(
            path_opa_home, "input", "split", "original.variables", "chl.test.txt"
        ),
    },
    {
        "name": "po4",
        "variable_name": "po4",
        "threshold_key": "N1p",
        "source_list": os.path.join(
            path_opa_home, "input", "split", "original.variables", "po4.test.txt"
        ),
    },
    {
        "name": "no3",
        "variable_name": "no3",
        "threshold_key": "N3n",
        "source_list": os.path.join(
            path_opa_home, "input", "split", "original.variables", "no3.test.txt"
        ),
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


def read_file_list(file_list_path):
    with open(file_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def make_link(source_path, link_path):
    source_path = os.path.abspath(source_path)
    if os.path.lexists(link_path):
        if os.path.islink(link_path) and os.path.abspath(os.readlink(link_path)) == source_path:
            return
        raise FileExistsError(f"Path already exists and is not the expected link: {link_path}")
    os.symlink(source_path, link_path)


def link_from_file_list(file_list_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    source_files = read_file_list(file_list_path)
    for source_file in source_files:
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
    with open(conf_path, "w") as f:
        json.dump(conf, f, indent=4)
        f.write("\n")
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
    domain_path = os.path.join(path_predictions_v2, domain_name)
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


if __name__ == "__main__":
    if not os.path.isdir(path_predictions):
        raise FileNotFoundError(f"Predictions folder not found: {path_predictions}")
    if not os.path.isdir(path_opa_home):
        raise FileNotFoundError(f"OPA_HOME folder not found: {path_opa_home}")

    Path(path_predictions_v2).mkdir(parents=True, exist_ok=True)

    process_domain("target", target_entries, source_kind="file_list")
    process_domain("input", input_entries, source_kind="file_list")
    process_domain("normal.net", normal_net_entries, source_kind="folder")
    process_domain("log.net", log_net_entries, source_kind="folder", already_log=True)
