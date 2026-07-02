import json
import os
import subprocess


path_predictions = "/leonardo_scratch/large/userexternal/gzuccari/TEST/predictions"
path_converted = os.path.join(path_predictions, "converted.variables")
path_original_postproc = os.path.join(path_predictions, "original.variables.postproc")
path_treshold = os.path.join(path_original_postproc, "treshold")
path_treshold_log = os.path.join(path_original_postproc, "treshold.log")

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

log_prediction_variables = {
    "chl.log.Chla.log": "Chla",
    "no3.log.N3n.log": "N3n",
    "po4.log.N1p.log": "N1p",
}

original_prediction_variables = {
    "chl.Chla": {"variable_name": "Chla", "treshold": 0.00001},
    "no3.N3n": {"variable_name": "N3n", "treshold": 0.001},
    "po4.N1p": {"variable_name": "N1p", "treshold": 0.00001},
}


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


def write_conf(conf_path, conf):
    with open(conf_path, "w") as f:
        json.dump(conf, f, indent=4)

    print("Configuration file created:")
    print(f"    {conf_path}")


def remove_conf(conf_path):
    os.remove(conf_path)
    print("Configuration file removed:")
    print(f"    {conf_path}")


if not os.path.isdir(path_predictions):
    raise FileNotFoundError(f"Predictions folder not found: {path_predictions}")

os.makedirs(path_converted, exist_ok=True)
os.makedirs(path_treshold, exist_ok=True)
os.makedirs(path_treshold_log, exist_ok=True)

for prediction_folder_name, variable_name in log_prediction_variables.items():
    folder_path = os.path.join(path_predictions, prediction_folder_name, variable_name)
    output_path = os.path.join(path_converted, prediction_folder_name)

    if not os.path.isdir(folder_path):
        print(f"Warning: input folder not found, skipping: {folder_path}")
        continue

    os.makedirs(output_path, exist_ok=True)

    conf_conversion = {
        "folder_path": folder_path,
        "output_path": output_path,
        "conversion_type": "exp",
        "variable_name": variable_name,
    }

    conf_conversion_path = os.path.join(
        path_converted,
        f"conf.exp.{prediction_folder_name}.json",
    )
    write_conf(conf_conversion_path, conf_conversion)
    run_mpi(converter_path, conf_conversion_path)
    remove_conf(conf_conversion_path)

for prediction_folder_name, variable_conf in original_prediction_variables.items():
    variable_name = variable_conf["variable_name"]
    treshold = variable_conf["treshold"]
    folder_path = os.path.join(path_predictions, prediction_folder_name, variable_name)
    treshold_output_path = os.path.join(path_treshold, prediction_folder_name)
    treshold_log_output_path = os.path.join(path_treshold_log, prediction_folder_name)

    if not os.path.isdir(folder_path):
        print(f"Warning: input folder not found, skipping: {folder_path}")
        continue

    os.makedirs(treshold_output_path, exist_ok=True)
    os.makedirs(treshold_log_output_path, exist_ok=True)

    conf_treshold = {
        "folder_path": folder_path,
        "output_path": treshold_output_path,
        "variable_name": variable_name,
        "treshold": treshold,
    }

    conf_treshold_path = os.path.join(
        path_treshold,
        f"conf.treshold.{prediction_folder_name}.json",
    )
    write_conf(conf_treshold_path, conf_treshold)
    run_mpi(apply_treshold_path, conf_treshold_path)
    remove_conf(conf_treshold_path)

    conf_conversion = {
        "folder_path": treshold_output_path,
        "output_path": treshold_log_output_path,
        "conversion_type": "log",
        "variable_name": variable_name,
    }

    conf_conversion_path = os.path.join(
        path_treshold_log,
        f"conf.treshold.log.{prediction_folder_name}.json",
    )
    write_conf(conf_conversion_path, conf_conversion)
    run_mpi(converter_path, conf_conversion_path)
    remove_conf(conf_conversion_path)
