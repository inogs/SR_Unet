import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Produce a torch river dataset from a config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf_make_rivers.json"),
        help="Path to configuration file.",
    )
    return parser.parse_args()


def validate_conf_file_path(conf_path):
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")
    if not os.path.isfile(conf_path):
        raise ValueError(f"Configuration path is not a file: {conf_path}")
    print("Configuration path check: passed")
    print(f"    Configuration path: {conf_path}")


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def read_file_list(file_path):
    files = []
    with open(file_path, "r") as f:
        for line in f:
            file_path = line.strip()
            if file_path:
                files.append(file_path)
    return files


def validate_conf(conf):
    required_fields = [
        "input_path",
        "output_path",
        "label",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    for field_name in required_fields:
        field_value = getattr(conf, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"Configuration field must be a non-empty string: {field_name}")

    if not os.path.exists(conf.input_path):
        raise FileNotFoundError(f"Input path not found: {conf.input_path}")
    if not os.path.isfile(conf.input_path):
        raise ValueError(f"Input path is not a file list: {conf.input_path}")

    print("Configuration file check: passed")
    for field_name, field_value in sorted(vars(conf).items()):
        print(f"    {field_name}: {field_value}")


def get_river_vector(river_files, label):
    river_vectors = []

    for i, file_path in enumerate(river_files):
        river_vectors.append(np.loadtxt(file_path))
        print(f"    Loaded file {i+1}/{len(river_files)}: {file_path}, {label} river vector appended")

    return np.array(river_vectors)


def make_rivers_dataset(conf):
    river_files = read_file_list(conf.input_path)
    rivers = get_river_vector(river_files, conf.label)
    os.makedirs(conf.output_path, exist_ok=True)

    output_file_path = os.path.join(conf.output_path, f"rivers_{conf.label}.pt")
    print(f"Saving the pytorch river {conf.label} dataset...")
    torch.save(torch.Tensor(rivers), output_file_path)
    print(f"    Saved: {output_file_path}")


if __name__ == "__main__":
    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    validate_conf(conf)

    print(f"[make_pt_rivers for label '{conf.label}'] Starting execution")
    make_rivers_dataset(conf)
    print(f"[make_pt_rivers for label '{conf.label}'] Ending execution")
