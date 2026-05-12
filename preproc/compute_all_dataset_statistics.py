#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for all txt file lists from a config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "conf_all_stat.json",
        ),
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


def validate_conf(conf):
    required_fields = [
        "txt_root",
        "output_root",
        "compute_script",
        "jobs",
        "recursive",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    path_fields = ["txt_root", "output_root", "compute_script"]
    for field_name in path_fields:
        field_value = getattr(conf, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"Configuration field must be a non-empty string: {field_name}"
            )

    if not os.path.exists(conf.txt_root):
        raise FileNotFoundError(f"Input folder not found: {conf.txt_root}")
    if not os.path.isdir(conf.txt_root):
        raise ValueError(f"Input path is not a folder: {conf.txt_root}")
    if os.path.exists(conf.output_root) and not os.path.isdir(conf.output_root):
        raise ValueError(f"Output path is not a folder: {conf.output_root}")
    if not os.path.exists(conf.compute_script):
        raise FileNotFoundError(f"Compute script not found: {conf.compute_script}")
    if not os.path.isfile(conf.compute_script):
        raise ValueError(f"Compute script path is not a file: {conf.compute_script}")

    if not isinstance(conf.jobs, int) or conf.jobs <= 0:
        raise ValueError("Configuration field jobs must be a positive integer")
    if not isinstance(conf.recursive, bool):
        raise ValueError("Configuration field recursive must be a boolean")

    if hasattr(conf, "max_files") and conf.max_files is not None:
        if not isinstance(conf.max_files, int) or conf.max_files <= 0:
            raise ValueError(
                "Configuration field max_files must be null or a positive integer"
            )

    print("Configuration file check: passed")
    for field_name, field_value in sorted(vars(conf).items()):
        print(f"    {field_name}: {field_value}")


def find_txt_files(root: Path, recursive: bool):
    pattern = "**/*.txt" if recursive else "*.txt"
    txt_files = []

    for txt_path in sorted(root.glob(pattern)):
        if not txt_path.is_file():
            continue
        if txt_path.name == "log.txt":
            continue
        if txt_path.name.startswith("stat."):
            continue

        first_line = first_non_empty_line(txt_path)
        if first_line is None:
            continue
        if not first_line.endswith(".nc"):
            continue

        txt_files.append(txt_path)

    return txt_files


def first_non_empty_line(txt_path: Path):
    with txt_path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def variable_from_txt_name(txt_path: Path):
    return txt_path.name.split(".")[0]


def build_command(
    txt_path: Path,
    variable: str,
    compute_script: Path,
    output_root: Path,
    jobs: int,
    max_files: int | None,
):
    command_parts = [
        sys.executable,
        str(compute_script),
        "-dp",
        str(txt_path),
        "-v",
        variable,
        "-j",
        str(jobs),
        "-op",
        str(output_root),
    ]

    if max_files is not None:
        command_parts.extend(["-n", str(max_files)])

    return command_parts


def main():
    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    validate_conf(conf)

    txt_root = Path(conf.txt_root)
    output_root = Path(conf.output_root)
    compute_script = Path(conf.compute_script)
    max_files = getattr(conf, "max_files", None)

    txt_files = find_txt_files(txt_root, conf.recursive)

    if not txt_files:
        raise FileNotFoundError(f"No input txt files found in {txt_root}")

    print(f"Found {len(txt_files)} txt files in {txt_root}")

    for txt_path in txt_files:
        variable = variable_from_txt_name(txt_path)
        command = build_command(
            txt_path=txt_path,
            variable=variable,
            compute_script=compute_script,
            output_root=output_root,
            jobs=conf.jobs,
            max_files=max_files,
        )

        print(f"Launching for {txt_path} with -v {variable}")
        subprocess.run(
            command,
            check=True,
            cwd=str(compute_script.parent),
        )


if __name__ == "__main__":
    main()
