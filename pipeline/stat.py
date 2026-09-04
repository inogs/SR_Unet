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
        description="Compute global mean and std for a file list from a configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(
            os.path.dirname(__file__), "conf.files.dir", "conf_stat.json"
        ),
        help="Path to configuration file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    if not os.path.isfile(args.config):
        raise ValueError(f"Configuration path is not a file: {args.config}")
    print("Configuration path check: passed")
    print(f"    Configuration path: {args.config}")

    return args


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def validate_conf(conf):
    required_fields = ["input_path", "output_path", "number_of_threads", "var_name"]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    if not os.path.exists(conf.input_path):
        raise ValueError(f"Input path does not exist: {conf.input_path}")
    if not os.path.isfile(conf.input_path):
        raise ValueError(f"Input path is not a file: {conf.input_path}")

    if not isinstance(conf.number_of_threads, int) or conf.number_of_threads <= 0:
        raise ValueError("Configuration field number_of_threads must be a positive integer")

    if not isinstance(conf.var_name, str) or not conf.var_name.strip():
        raise ValueError(f"Configuration field var_name must be a non-empty string: {conf.var_name}")

    print("Configuration file check: passed")
    for field_name, field_value in sorted(vars(conf).items()):
        print(f"    {field_name}: {field_value}")


def read_file_list(path):
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                files.append(line)
    return files


def init_worker_id(counter, lock):
    global WORKER_ID
    with lock:
        WORKER_ID = counter.value
        counter.value += 1


def file_stats(path, variable):
    with nc.Dataset(path) as ds:
        if variable not in ds.variables:
            raise KeyError(f"{variable} not found in {path}")

        nc_var = ds[variable]
        array = nc_var[:]

        mu = float(ma.mean(array))
        var = float(ma.var(array))
        data_min = float(ma.min(array))
        data_max = float(ma.max(array))

        if "depth" in nc_var.dimensions:
            idx = [slice(None)] * nc_var.ndim
            idx[nc_var.dimensions.index("depth")] = 0
            surface_array = array[tuple(idx)]
        else:
            surface_array = array

        surface_min = float(ma.min(surface_array))
        surface_max = float(ma.max(surface_array))

    return mu, var, data_min, data_max, surface_min, surface_max


def file_stats_logged(path, variable):
    worker_id = 0 if WORKER_ID is None else WORKER_ID
    print(f"[worker {worker_id}] processing {path}", flush=True)
    return file_stats(path, variable)


def combine_equal_mask(mus, vars_):
    mus = np.asarray(mus, dtype=np.float64)
    vars_ = np.asarray(vars_, dtype=np.float64)

    mu_tot = mus.mean()
    sigma = np.sqrt(np.mean(vars_ + (mus - mu_tot) ** 2))

    return mu_tot, sigma

def compute_all_file_stats(files, variable, jobs):
    """
    Calcola UNA VOLTA mean e variance di ogni NetCDF.
    Restituisce due array con una posizione per file.
    """

    if jobs == 1:
        stats = [
            file_stats(f, variable)
            for f in files
        ]
    else:
        with Manager() as manager:
            counter = manager.Value("i", 0)
            lock = manager.Lock()

            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=init_worker_id,
                initargs=(counter, lock),
            ) as ex:

                stats = list(
                    ex.map(
                        file_stats_logged,
                        files,
                        [variable] * len(files)
                    )
                )

    mus = np.array(
        [s[0] for s in stats],
        dtype=np.float64
    )

    vars_ = np.array(
        [s[1] for s in stats],
        dtype=np.float64
    )

    return mus, vars_


def compute_stats_from_cached_files(mus, vars_, indices):
    """
    Combina SOLO le statistiche dei file appartenenti
    al training fold.
    """

    fold_mus = mus[indices]
    fold_vars = vars_[indices]

    return combine_equal_mask(
        fold_mus,
        fold_vars
    )


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

    mus, vars_, mins, maxs, surface_mins, surface_maxs = zip(*stats)

    print(f"Computed per-file stats in {time.time() - t0:.2f} s")

    # --- combine ---
    t0 = time.time()
    mu_tot, sigma = combine_equal_mask(mus, vars_)
    max_global = float(np.max(maxs))
    min_global = float(np.min(mins))
    max_surface = float(np.max(surface_maxs))
    min_surface = float(np.min(surface_mins))
    print(f"Reduction done in {time.time() - t0:.4f} s")

    print(f"Mean         = {mu_tot:.10e}")
    print(f"Std          = {sigma:.10e}")
    print(f"Max (global) = {max_global:.10e}")
    print(f"Min (global) = {min_global:.10e}")
    print(f"Max (surface) = {max_surface:.10e}")
    print(f"Min (surface) = {min_surface:.10e}")

    return mu_tot, sigma, max_global, min_global, max_surface, min_surface


def write_stats(out_dir, name, mu_tot, sigma):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stat.{name}.txt"

    with open(out_path, "w") as f:
        f.write(f"{mu_tot:.16e}\n")
        f.write(f"{sigma:.16e}\n")

    print(f"Saved results to {out_path}")


def write_stats_json(out_dir, name, mu_tot, sigma, max_global, min_global, max_surface, min_surface):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stat.{name}.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "mean": mu_tot,
                "std": sigma,
                "max_global": max_global,
                "min_global": min_global,
                "max_surface": max_surface,
                "min_surface": min_surface,
            },
            f,
            indent=4,
        )
        f.write("\n")

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_conf_file(args.config)
    validate_conf(conf)

    mu_tot, sigma, max_global, min_global, max_surface, min_surface = compute_list_stats(
        conf.input_path, conf.var_name, conf.number_of_threads
    )
    name = Path(conf.input_path).stem
    write_stats(conf.output_path, name, mu_tot, sigma)
    write_stats_json(
        conf.output_path, name, mu_tot, sigma, max_global, min_global, max_surface, min_surface
    )
