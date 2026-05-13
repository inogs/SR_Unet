import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from types import SimpleNamespace
import netCDF4 as nc
import numpy as np
import numpy.ma as ma


WORKER_ID = None


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Compute global mean and std for all txt file lists in a folder."
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Path to JSON configuration file")
    return parser.parse_args()


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def validate_conf(conf):
    required_fields = ["input_path", "output_path", "jobs"]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    for field_name in ["input_path", "output_path"]:
        field_value = getattr(conf, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"Configuration field must be a non-empty string: {field_name}"
            )

    if not os.path.exists(conf.input_path):
        raise FileNotFoundError(f"Input folder not found: {conf.input_path}")
    if not os.path.isdir(conf.input_path):
        raise ValueError(f"Input path is not a folder: {conf.input_path}")
    if os.path.exists(conf.output_path) and not os.path.isdir(conf.output_path):
        raise ValueError(f"Output path is not a folder: {conf.output_path}")

    if not isinstance(conf.jobs, int) or conf.jobs <= 0:
        raise ValueError("Configuration field jobs must be a positive integer")
    if hasattr(conf, "recursive") and not isinstance(conf.recursive, bool):
        raise ValueError("Configuration field recursive must be a boolean")


def read_file_list(path):
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                files.append(line)
    return files


def first_non_empty_line(txt_path):
    with txt_path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def find_txt_files(root, recursive=False):
    pattern = "**/*.txt" if recursive else "*.txt"
    txt_files = []

    for txt_path in sorted(root.glob(pattern)):
        if not txt_path.is_file():
            continue
        if txt_path.name == "log.txt":
            continue
        if txt_path.name.startswith("stat."):
            continue
        if first_non_empty_line(txt_path) is None:
            continue

        txt_files.append(txt_path)

    return txt_files


def variable_from_txt_name(txt_path):
    return txt_path.name.split(".")[0]


def file_stats(path, variable):
    with nc.Dataset(path) as ds:
        if variable not in ds.variables:
            raise KeyError(f"{variable} not found in {path}")

        array = ds[variable][:]

        mu = float(ma.mean(array))
        var = float(ma.var(array))

    return mu, var


def init_worker_id(counter, lock):
    global WORKER_ID
    with lock:
        WORKER_ID = counter.value
        counter.value += 1


def file_stats_logged(path, variable):
    worker_id = 0 if WORKER_ID is None else WORKER_ID
    print(f"[worker {worker_id}] processing {path}", flush=True)
    return file_stats(path, variable)


def combine_equal_mask(mus, vars_):
    mus = np.asarray(mus, dtype=np.float64)
    vars_ = np.asarray(vars_, dtype=np.float64)

    mu_tot = mus.mean()
    sigma = np.sqrt(np.mean(vars_ + (mus - mu_tot)**2))

    return mu_tot, sigma


def compute_list_stats(txt_path, variable, jobs):
    t0 = time.time()
    files = read_file_list(txt_path)
    if not files:
        raise ValueError(f"No files found in list: {txt_path}")

    print(f"Loaded {len(files)} files from {txt_path} in {time.time() - t0:.2f} s")

    t0 = time.time()

    if jobs == 1:
        stats = [file_stats(f, variable) for f in files]
    else:
        with Manager() as manager:
            counter = manager.Value("i", 0)
            lock = manager.Lock()
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=init_worker_id,
                initargs=(counter, lock),
            ) as ex:
                stats = list(ex.map(
                    file_stats_logged,
                    files,
                    [variable] * len(files)
                ))

    mus, vars_ = zip(*stats)

    print(f"Computed per-file stats in {time.time() - t0:.2f} s")

    # --- combine ---
    t0 = time.time()
    mu_tot, sigma = combine_equal_mask(mus, vars_)
    print(f"Reduction done in {time.time() - t0:.4f} s")

    print(f"Mean  = {mu_tot:.10e}")
    print(f"Std   = {sigma:.10e}")

    return mu_tot, sigma


def write_stats(out_dir, txt_path, mu_tot, sigma):
    name = txt_path.stem
    out_path = out_dir / f"stat.{name}.txt"

    with open(out_path, "w") as f:
        f.write(f"{mu_tot:.16e}\n")
        f.write(f"{sigma:.16e}\n")

    print(f"Saved results to {out_path}")


def main():
    args = parse_input_parameters()
    conf = read_conf_file(args.config)
    validate_conf(conf)

    input_path = Path(conf.input_path)
    output_path = Path(conf.output_path)
    recursive = getattr(conf, "recursive", False)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = find_txt_files(input_path, recursive)
    if not txt_files:
        raise FileNotFoundError(f"No input txt files found in {input_path}")

    print(f"Found {len(txt_files)} txt files in {input_path}")

    for txt_path in txt_files:
        variable = variable_from_txt_name(txt_path)
        print(f"Computing {txt_path} with variable {variable}")
        mu_tot, sigma = compute_list_stats(txt_path, variable, conf.jobs)
        write_stats(output_path, txt_path, mu_tot, sigma)


if __name__ == "__main__":
    main()
