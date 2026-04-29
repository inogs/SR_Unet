import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import numpy as np
import xarray as xr


WORKER_ID = None


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Compute global mean and std from multiple NetCDF files."
    )
    parser.add_argument("-dp", "--file-path", required=True,
                        help="Path to text file containing list of NetCDF files")
    parser.add_argument("-v", "--variable", required=True,
                        help="Variable name inside NetCDF files")
    parser.add_argument("-n", type=int, default=None,
                        help="Max number of files to process")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Number of parallel workers (default=1 = serial)")

    args = parser.parse_args()

    if args.n is not None and args.n <= 0:
        raise ValueError("-n must be positive")
    if args.jobs <= 0:
        raise ValueError("-j must be positive")

    return args


def read_file_list(path, n=None):
    files = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            line = line.strip()
            if line:
                files.append(line)
    return files


def file_stats(path, variable):
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable} not found in {path}")

        x = ds[variable]

        mu = x.mean(skipna=True).item()
        var = x.var(skipna=True).item()

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


def main():
    args = parse_input_parameters()

    # --- read file list ---
    t0 = time.time()
    files = read_file_list(args.file_path, args.n)
    print(f"Loaded {len(files)} files in {time.time() - t0:.2f} s")

    # --- compute per-file stats ---
    t0 = time.time()

    if args.jobs == 1:
        stats = [file_stats(f, args.variable) for f in files]
    else:
        with Manager() as manager:
            counter = manager.Value("i", 0)
            lock = manager.Lock()
            with ProcessPoolExecutor(
                max_workers=args.jobs,
                initializer=init_worker_id,
                initargs=(counter, lock),
            ) as ex:
                stats = list(ex.map(
                    file_stats_logged,
                    files,
                    [args.variable] * len(files)
                ))

    mus, vars_ = zip(*stats)

    print(f"Computed per-file stats in {time.time() - t0:.2f} s")

    # --- combine ---
    t0 = time.time()
    mu_tot, sigma = combine_equal_mask(mus, vars_)
    print(f"Reduction done in {time.time() - t0:.4f} s")

    print(f"Mean  = {mu_tot:.10e}")
    print(f"Std   = {sigma:.10e}")

    # --- write output ---
    base_name = os.path.basename(args.file_path)
    name = os.path.splitext(base_name)[0]
    out_path = os.path.join(
        os.path.dirname(args.file_path),
        f"stat.{name}.txt"
    )

    with open(out_path, "w") as f:
        f.write(f"{mu_tot:.16e}\n")
        f.write(f"{sigma:.16e}\n")

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()