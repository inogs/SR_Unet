import json
import os
import subprocess


path_predictions = "/leonardo_scratch/large/userexternal/gzuccari/TEST/predictions"
path_converted = os.path.join(path_predictions, "converted.variables")
# create path_converted if it does not exist
if not os.path.isdir(path_converted):
    os.makedirs(path_converted, exist_ok=True)
number_of_processes = 16

converter_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "pipeline",
    "converter_mpi.py",
)

log_prediction_variables = {
    "chl.log.Chla.log": "Chla",
    "no3.log.N3n.log": "N3n",
    "po4.log.N1p.log": "N1p",
}


if not os.path.isdir(path_predictions):
    raise FileNotFoundError(f"Predictions folder not found: {path_predictions}")

os.makedirs(path_converted, exist_ok=True)

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
    with open(conf_conversion_path, "w") as f:
        json.dump(conf_conversion, f, indent=4)

    print(f"Configuration file created for {prediction_folder_name}:")
    print(f"    {conf_conversion_path}")

    cmd = [
        "mpirun",
        "-np",
        str(number_of_processes),
        "python",
        converter_path,
        "--config",
        conf_conversion_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(conf_conversion_path)
    print(f"Configuration file removed for {prediction_folder_name}")
