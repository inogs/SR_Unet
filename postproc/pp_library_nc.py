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

def file_field_logged(path, variable):
    worker_id = 0 if WORKER_ID is None else WORKER_ID
    print(f"[worker {worker_id}] processing {path}", flush=True)
    return file_field(path, variable)

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

def file_field(path, variable):
    with nc.Dataset(path) as ds:
        if variable not in ds.variables:
            raise KeyError(f"{variable} not found in {path}")
        return ma.array(ds.variables[variable][:], dtype=np.float64)

def combine_equal_mask(mus, vars_):
    mus = np.asarray(mus, dtype=np.float64)
    vars_ = np.asarray(vars_, dtype=np.float64)

    mu_tot = mus.mean()
    sigma = np.mean(np.sqrt(vars_))

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

def compute_list_field_stats(txt_path, variable, jobs):
    files = read_file_list(txt_path)
    if not files:
        raise ValueError(f"No files found in list: {txt_path}")

    if jobs == 1:
        arrays = [file_field(file_path, variable) for file_path in files]
    else:
        with Manager() as manager:
            counter = manager.Value("i", 0)
            lock = manager.Lock()
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=init_worker_id,
                initargs=(counter, lock),
            ) as ex:
                arrays = list(ex.map(
                    file_field_logged,
                    files,
                    [variable] * len(files)
                ))

    stack = ma.stack(arrays, axis=0)
    return ma.mean(stack, axis=0), ma.std(stack, axis=0), files[0]

def write_field(reference_file, output_path, variable, field):
    with nc.Dataset(reference_file) as src, nc.Dataset(output_path, "w", format="NETCDF4") as dst:
        for attr_name in src.ncattrs():
            dst.setncattr(attr_name, src.getncattr(attr_name))

        for dim_name, dimension in src.dimensions.items():
            dst.createDimension(
                dim_name,
                len(dimension) if not dimension.isunlimited() else None,
            )

        for var_name, src_var in src.variables.items():
            if "_FillValue" in src_var.ncattrs():
                dst_var = dst.createVariable(
                    var_name,
                    src_var.datatype,
                    src_var.dimensions,
                    fill_value=src_var.getncattr("_FillValue"),
                )
            else:
                dst_var = dst.createVariable(
                    var_name,
                    src_var.datatype,
                    src_var.dimensions,
                )

            for attr_name in src_var.ncattrs():
                if attr_name != "_FillValue":
                    dst_var.setncattr(attr_name, src_var.getncattr(attr_name))

            if var_name == variable:
                dst_var[:] = field
                if field.count() > 0:
                    data_min = float(ma.min(field))
                    data_max = float(ma.max(field))
                    if "valid_min" in dst_var.ncattrs():
                        dst_var.setncattr("valid_min", data_min)
                    if "valid_max" in dst_var.ncattrs():
                        dst_var.setncattr("valid_max", data_max)
                    if "actual_range" in dst_var.ncattrs():
                        dst_var.setncattr(
                            "actual_range",
                            np.array([data_min, data_max], dtype=np.float32),
                        )
            else:
                dst_var[:] = src_var[:]

    print(f"Saved field to {output_path}")
