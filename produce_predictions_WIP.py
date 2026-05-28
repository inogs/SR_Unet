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
        "stats_files",
    ]
    for key in required:
        if key not in conf:
            raise ValueError(f"Missing configuration field: {key}")

    check_file(conf["weight_file"], "weight_file")
    check_file(conf["test_path"], "test_path")
    check_file(conf.get("river_path"), "river_path")
    check_output_path(conf["output_path"])

    for var in conf["variables"]:
        if var not in conf["reference_nc_lists"]:
            raise ValueError(f"Missing reference_nc_lists entry for variable: {var}")
        if var not in conf["stats_files"]:
            raise ValueError(f"Missing stats_files entry for variable: {var}")

        check_file(conf["reference_nc_lists"][var], f"reference_nc_lists[{var}]")
        check_file(conf["stats_files"][var], f"stats_files[{var}]")
        for nc_path in read_file_list(conf["reference_nc_lists"][var]):
            check_file(nc_path, f"reference NetCDF for {var}")

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
            values = [np.float32(line.strip()) for line in file if line.strip()]
        if len(values) < 2:
            raise ValueError(f"Stats file must contain mean and std: {conf['stats_files'][var]}")
        stats[var] = (values[0], values[1])
    return stats


def load_reference_files(conf, n_samples):
    reference_files = {}
    for var in conf["variables"]:
        files = read_file_list(conf["reference_nc_lists"][var])
        if len(files) != n_samples:
            raise ValueError(f"Reference files for {var}: {len(files)} != {n_samples} test samples")
        reference_files[var] = files
    return reference_files


def prediction_filename(var, reference_file):
    filename = os.path.basename(reference_file)
    if filename.startswith(var):
        return var + filename[len(var):].replace("-", "_")
    return f"{var}_{filename}".replace("-", "_")


def write_prediction(reference_file, output_path, var, prediction):
    var_output_path = os.path.join(output_path, var)
    os.makedirs(var_output_path, exist_ok=True)
    destination_path = os.path.join(var_output_path, prediction_filename(var, reference_file))

    prediction = prediction.detach().cpu().numpy()
    with nc.Dataset(reference_file) as source, nc.Dataset(destination_path, "w", format="NETCDF4") as dest:
        source_var = source.variables[var]

        for dim_name in source_var.dimensions:
            source_dim = source.dimensions[dim_name]
            dest.createDimension(dim_name, None if source_dim.isunlimited() else len(source_dim))
            if dim_name in source.variables:
                source_coord = source.variables[dim_name]
                dest_coord = dest.createVariable(dim_name, source_coord.dtype, source_coord.dimensions)
                dest_coord[:] = source_coord[:]

        fill_value = getattr(source_var, "_FillValue", None)
        kwargs = {"fill_value": fill_value} if fill_value is not None else {}
        dest_var = dest.createVariable(var, source_var.dtype, source_var.dimensions, **kwargs)
        dest_var[:] = ma.masked_array(prediction, mask=ma.getmaskarray(source_var[:]))

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

    os.makedirs(conf["output_path"], exist_ok=True)
    with torch.no_grad():
        for start in range(0, x.shape[0], conf["batch_size"]):
            stop = min(start + conf["batch_size"], x.shape[0])
            batch = x[start:stop].clone()
            batch[batch > conf["mask_threshold"]] = 0
            batch = batch.to(device)

            if rivers is None:
                prediction = model(batch)
            else:
                prediction = model(batch, rivers[start:stop].to(device))

            for sample_offset, sample_index in enumerate(range(start, stop)):
                for var_index, var in enumerate(conf["variables"]):
                    mean, std = stats[var]
                    pred_var = prediction[sample_offset, var_index] * std + mean
                    write_prediction(
                        reference_files[var][sample_index],
                        conf["output_path"],
                        var,
                        pred_var,
                    )


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
