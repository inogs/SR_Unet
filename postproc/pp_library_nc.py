import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from types import SimpleNamespace
import netCDF4 as nc
import numpy as np
import numpy.ma as ma

from pp_library import ( read_file_list )

def init_worker_id(counter, lock):
    global WORKER_ID
    with lock:
        WORKER_ID = counter.value
        counter.value += 1

def write_stats(out_dir, name, mu_tot, sigma):
    out_dir = Path(out_dir)
    out_path = out_dir / f"stat.{name}.txt"

    with open(out_path, "w") as f:
        f.write(f"{mu_tot:.16e}\n")
        f.write(f"{sigma:.16e}\n")

    print(f"Saved results to {out_path}")

def file_stats_logged(path, variable):
    worker_id = 0 if WORKER_ID is None else WORKER_ID
    print(f"[worker {worker_id}] processing {path}", flush=True)
    return file_stats(path, variable)

def file_stats(path, variable):
    with nc.Dataset(path) as ds:
        if variable not in ds.variables:
            raise KeyError(f"{variable} not found in {path}")

        var = ds[variable]
        if "depth" in var.dimensions:
            idx = [slice(None)] * var.ndim
            idx[var.dimensions.index("depth")] = 0
            array = var[tuple(idx)]
        else:
            array = var[:]

        mu = float(ma.mean(array))
        var = float(ma.var(array))

    return mu, var

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