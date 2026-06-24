import argparse
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

def init_worker_id(counter, lock):
    global WORKER_ID
    with lock:
        WORKER_ID = counter.value
        counter.value += 1

def extract_year_and_season(s):
    file_name = os.path.basename(s)
    groups = re.findall(r"\d+", file_name)
    if len(groups) < 2:
        return (None, None)
    return groups[-2], groups[-1]

def determine_season(season_number):
    t = int(season_number)
    if 0 <= t < 18:
        return "winter"
    elif 18 <= t < 36:
        return "spring"
    elif 36 <= t < 54:
        return "summer"
    elif 54 <= t <= 72:
        return "autumn"
    else:
        return None

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

# def file_stats(path, variable):
#     with nc.Dataset(path) as ds:
#         if variable not in ds.variables:
#             raise KeyError(f"{variable} not found in {path}")

#         array = ds[variable][:]

#         mu = float(ma.mean(array))
#         var = float(ma.var(array))

#     return mu, var

def read_file_list(path):
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                files.append(line)
    return files

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


path_output = os.path.join(os.path.dirname(__file__), "out")
path_input = os.path.join(os.path.dirname(__file__), "Chla.test.txt")
print(path_output)
print(path_input)


list_input_files = []
list_output_winter_files = []
list_output_spring_files = []
list_output_summer_files = []
list_output_autumn_files = []

# read from input
with open(path_input, "r") as f:
    for line in f:
        list_input_files.append(line.strip())

# print first 5 items, line by line
print("First 5 items in the input file:")
for item in list_input_files[:5]:
    print(item)


list_years = []
list_seasons = []
for file_name in list_input_files:
    year, season = extract_year_and_season(file_name)
    list_years.append(year)
    list_seasons.append(season)
    season_name = determine_season(season)
    if season_name == "winter":
        list_output_winter_files.append(file_name)
    elif season_name == "spring":
        list_output_spring_files.append(file_name)
    elif season_name == "summer":
        list_output_summer_files.append(file_name)
    elif season_name == "autumn":
        list_output_autumn_files.append(file_name)

print("Years extracted from input files:")
print(list_years)
print("Seasons extracted from input files:")
print(list_seasons)

print("Number of files in each season:")
print(f"Winter: {len(list_output_winter_files)}")
print(f"Spring: {len(list_output_spring_files)}")
print(f"Summer: {len(list_output_summer_files)}")
print(f"Autumn: {len(list_output_autumn_files)}")

# write the output files
with open(os.path.join(path_output, "winter.txt"), "w") as f:
    for item in list_output_winter_files:
        f.write("%s\n" % item)
with open(os.path.join(path_output, "spring.txt"), "w") as f:
    for item in list_output_spring_files:
        f.write("%s\n" % item)
with open(os.path.join(path_output, "summer.txt"), "w") as f:
    for item in list_output_summer_files:
        f.write("%s\n" % item)
with open(os.path.join(path_output, "autumn.txt"), "w") as f:
    for item in list_output_autumn_files:
        f.write("%s\n" % item)

path_winter_files = os.path.join(path_output, "winter.txt")
jobs = 16

mu_winter, sigma_winter = compute_list_stats(path_winter_files, "Chla", jobs)

print(f"Winter: Mean = {mu_winter:.10e}, Std = {sigma_winter:.10e}")
write_stats(path_output, "Chla.test.winter", mu_winter, sigma_winter)
