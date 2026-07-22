import argparse
import json
import os
import time

import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pytorch_lightning as pl
import torch

from models.convolutional.conv_model import ConvModel


def parse_args():
    parser = argparse.ArgumentParser(description="Produce SR-UNet predictions.")
    parser.add_argument("-c", "--config", required=True, help="Prediction configuration JSON.")
    return parser.parse_args()


def read_conf(conf_path):
    with open(conf_path, "r") as file:
        return json.load(file)


def check_file(path, name):
    if path is None:
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def check_output_path(path):
    parent = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"output_path parent not found: {parent}")
    if os.path.exists(path) and not os.path.isdir(path):
        raise ValueError(f"output_path exists but is not a directory: {path}")


def check_output_subdir(path):
    if path is None:
        return
    if not isinstance(path, str) or not path.strip():
        raise ValueError("prediction_subdir must be a non-empty string")
    if os.path.isabs(path) or ".." in path.split(os.sep):
        raise ValueError("prediction_subdir must be relative and cannot contain '..'")


def read_file_list(list_path):
    with open(list_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


def validate_conf(conf):
    required = [
        "weight_file",
        "test_path",
        "output_path",
        "variables",
        "reference_nc_lists",
        "reference_target_nc_lists",
        "target_variables",
        "stats_files",
    ]
    for key in required:
        if key not in conf:
            raise ValueError(f"Missing configuration field: {key}")

    check_file(conf["weight_file"], "weight_file")
    check_file(conf["test_path"], "test_path")
    check_file(conf.get("river_path"), "river_path")
    check_output_path(conf["output_path"])
    check_output_subdir(conf.get("prediction_subdir"))

    for var in conf["variables"]:
        if var not in conf["reference_nc_lists"]:
            raise ValueError(f"Missing reference_nc_lists entry for variable: {var}")
        if var not in conf["target_variables"]:
            raise ValueError(f"Missing target_variables entry for variable: {var}")
        if var not in conf["stats_files"]:
            raise ValueError(f"Missing stats_files entry for variable: {var}")

        target_var = conf["target_variables"][var]
        if target_var not in conf["reference_target_nc_lists"]:
            raise ValueError(f"Missing reference_target_nc_lists entry for target variable: {target_var}")

        check_file(conf["reference_nc_lists"][var], f"reference_nc_lists[{var}]")
        check_file(conf["reference_target_nc_lists"][target_var], f"reference_target_nc_lists[{target_var}]")
        check_file(conf["stats_files"][var], f"stats_files[{var}]")
        for nc_path in read_file_list(conf["reference_nc_lists"][var]):
            check_file(nc_path, f"reference NetCDF for {var}")
        for nc_path in read_file_list(conf["reference_target_nc_lists"][target_var]):
            check_file(nc_path, f"reference target NetCDF for {target_var}")

    conf.setdefault("batch_size", 1)
    conf.setdefault("mask_threshold", 1000000.0)
    conf.setdefault("device", "auto")
    conf.setdefault("seed", 0)

    if conf["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

    print("Configuration check: passed")
    for key, value in sorted(conf.items()):
        print(f"    {key}: {value}")


def get_device(conf):
    if conf["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(conf["device"])


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_test_input(test_path):
    try:
        torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])
    except AttributeError:
        pass

    dataset = torch_load(test_path, map_location="cpu")
    if hasattr(dataset, "tensors"):
        x = dataset.tensors[0]
    elif torch.is_tensor(dataset):
        x = dataset
    else:
        x = torch.as_tensor(np.array(dataset)[:, 0])

    if x.ndim != 5:
        raise ValueError(f"Expected input shape (sample, channel, depth, lat, lon), found {tuple(x.shape)}")
    return x.float()


def load_rivers(river_path, n_samples):
    if river_path is None:
        return None

    rivers = torch_load(river_path, map_location="cpu")
    if not torch.is_tensor(rivers):
        rivers = torch.as_tensor(rivers)
    if rivers.shape[0] != n_samples:
        raise ValueError(f"River samples do not match test samples: {rivers.shape[0]} != {n_samples}")
    return rivers.float()


def load_stats(conf):
    stats = {}
    for var in conf["variables"]:
        with open(conf["stats_files"][var], "r") as file:
            data = json.load(file)
        stats[var] = (np.float32(data["mean"]), np.float32(data["std"]))
    return stats


def load_reference_files(conf, n_samples):
    reference_files = {}
    for var in conf["variables"]:
        target_var = conf["target_variables"][var]
        files = read_file_list(conf["reference_target_nc_lists"][target_var])
        if len(files) != n_samples:
            raise ValueError(f"Reference target files for {target_var}: {len(files)} != {n_samples} test samples")
        reference_files[var] = files
    return reference_files


def prediction_filename(var, reference_file):
    filename = os.path.basename(reference_file)
    if filename.startswith(var):
        return var + filename[len(var):].replace("-", "_")
    return f"{var}_{filename}".replace("-", "_")


def write_prediction(reference_file, output_path, target_var, prediction):
    var_output_path = os.path.join(output_path, target_var)
    os.makedirs(var_output_path, exist_ok=True)
    destination_path = os.path.join(var_output_path, prediction_filename(target_var, reference_file))

    prediction = prediction.detach().cpu().numpy()
    with nc.Dataset(reference_file) as source, nc.Dataset(destination_path, "w", format="NETCDF4") as dest:
        source_var = source.variables[target_var]

        for attr_name in source.ncattrs():
            dest.setncattr(attr_name, source.getncattr(attr_name))

        for dim_name, dimension in source.dimensions.items():
            dest.createDimension(
                dim_name,
                len(dimension) if not dimension.isunlimited() else None,
            )

        for var_name, src_var in source.variables.items():
            if "_FillValue" in src_var.ncattrs():
                dest_var = dest.createVariable(
                    var_name,
                    src_var.datatype,
                    src_var.dimensions,
                    fill_value=src_var.getncattr("_FillValue"),
                )
            else:
                dest_var = dest.createVariable(
                    var_name,
                    src_var.datatype,
                    src_var.dimensions,
                )

            for attr_name in src_var.ncattrs():
                if attr_name != "_FillValue":
                    dest_var.setncattr(attr_name, src_var.getncattr(attr_name))

            if var_name == target_var:
                masked_prediction = ma.masked_array(prediction, mask=ma.getmaskarray(source_var[:]))
                dest_var[:] = masked_prediction
                if masked_prediction.count() > 0:
                    data_min = float(ma.min(masked_prediction))
                    data_max = float(ma.max(masked_prediction))
                    if "valid_min" in dest_var.ncattrs():
                        dest_var.setncattr("valid_min", data_min)
                    if "valid_max" in dest_var.ncattrs():
                        dest_var.setncattr("valid_max", data_max)
                    if "actual_range" in dest_var.ncattrs():
                        dest_var.setncattr(
                            "actual_range",
                            np.array([data_min, data_max], dtype=np.float32),
                        )
            else:
                dest_var[:] = src_var[:]

    print(destination_path)


def predict(conf):
    pl.seed_everything(conf["seed"], workers=True)
    device = get_device(conf)

    x = load_test_input(conf["test_path"])
    rivers = load_rivers(conf.get("river_path"), x.shape[0])
    stats = load_stats(conf)
    reference_files = load_reference_files(conf, x.shape[0])

    model = ConvModel.load_from_checkpoint(conf["weight_file"], map_location=device)
    model.eval()
    model.to(device)

    output_path = conf["output_path"]
    if conf.get("prediction_subdir"):
        output_path = os.path.join(output_path, conf["prediction_subdir"])

    os.makedirs(output_path, exist_ok=True)
    inference_time = 0.0
    with torch.no_grad():
        for start in range(0, x.shape[0], conf["batch_size"]):
            stop = min(start + conf["batch_size"], x.shape[0])
            batch = x[start:stop].clone()
            batch[batch > conf["mask_threshold"]] = 0
            batch = batch.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inference = time.time()

            if rivers is None:
                prediction = model(batch)
            else:
                prediction = model(batch, rivers[start:stop].to(device))

            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_time += time.time() - t_inference

            for sample_offset, sample_index in enumerate(range(start, stop)):
                for var_index, var in enumerate(conf["variables"]):
                    mean, std = stats[var]
                    target_var = conf["target_variables"][var]
                    pred_var = prediction[sample_offset, var_index] * std + mean
                    write_prediction(
                        reference_files[var][sample_index],
                        output_path,
                        target_var,
                        pred_var,
                    )

    test_name = os.path.basename(conf["test_path"])
    print(f"[prediction for dataset '{test_name}'] Inference time: {inference_time:.2f} seconds")


if __name__ == "__main__":
    args = parse_args()
    conf = read_conf(args.config)
    validate_conf(conf)

    test_name = os.path.basename(conf["test_path"])
    t0 = time.time()
    print(f"[prediction for dataset '{test_name}'] Starting execution")
    predict(conf)
    print(f"[prediction for dataset '{test_name}'] Ending execution")
    print(f"[prediction for dataset '{test_name}'] Total time taken: {time.time() - t0:.2f} seconds")
