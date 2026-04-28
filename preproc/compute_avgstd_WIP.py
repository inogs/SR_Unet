import argparse
import os
import resource
import time
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import xarray as xr


def _rss_mb_linux():
    try:
        with open("/proc/self/status", "r") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _array_memory_mb(array):
    if isinstance(array, ma.MaskedArray):
        data_bytes = array.data.nbytes
        mask = array.mask
        mask_bytes = 0 if mask is ma.nomask else mask.nbytes
    else:
        data_bytes = array.nbytes
        mask_bytes = 0
    return data_bytes / (1024 ** 2), mask_bytes / (1024 ** 2)


def _format_mb(value):
    if value is None:
        return "NA"
    return f"{value:.2f}"


def _log_memory(tag, baseline_mb=None, extra_info=""):
    rss_now_mb = _rss_mb_linux()
    rss_delta_mb = None if baseline_mb is None or rss_now_mb is None else rss_now_mb - baseline_mb
    prefix = f"[{tag}]"
    if extra_info:
        prefix = f"{prefix} {extra_info}"
    print(
        f"{prefix} rss_now_mb={_format_mb(rss_now_mb)} "
        f"rss_delta_mb={_format_mb(rss_delta_mb)} "
        f"rss_peak_mb={_format_mb(_peak_rss_mb())}"
    )

def parse_input_parameters():

    parser = argparse.ArgumentParser(
        description="Split the dataset into train and test files."
    )
    # parse following arguments:
    # -data_path (string): path of the data files (mandatory)
    # -variable name (string): variable to extract from the data files (mandatory)
    parser.add_argument("-dp", "--file-path", required=True, help="Path of the file list")
    parser.add_argument("-v", "--variable", required=True, help="Variable to extract from the data files")
    parser.add_argument("-n", type=int, default=None, help="Maximum number of files to process")

    args = parser.parse_args()    

    if args.n is not None and args.n <= 0:
        raise ValueError("-n must be a positive integer")

    return args

if __name__ == '__main__':

    print("Hello World!")
    args = parse_input_parameters()
    print(f"File path: {args.file_path}")
    print(f"Variable: {args.variable}")
    if args.n is not None:
        print(f"Max files to process: {args.n}")

    # open the file, read every line and store it in a list
    file_list = []
    with open(args.file_path, 'r') as f:
        file_list = f.readlines()
    # print first 5 elements and total number
    print(f"Total number of files: {len(file_list)}")
    # print(f"First 5 files: {file_list[:5]}")

    # 

    first_file = file_list[0].strip()
    print(f"First file: {first_file}")
    with nc.Dataset(first_file) as file_ds:
        print(f"Variables in the file: {file_ds.variables.keys()}")

        # get variables list
        var_list = list(file_ds.variables.keys())
        print(f"Variables list: {var_list}")

        # check if the args.variable in in the var_list, return
        # error otherwise
        if args.variable not in var_list:
            raise ValueError(f"Variable {args.variable} not found in the file. Available variables: {var_list}")

        array = file_ds[args.variable][:]
    print(f"Shape of the variable array: {array.shape}")

    num_nan = np.isnan(array).sum()
    print(f"Number of NAN in the variable array: {num_nan}")
    

    # define list, cycle over the file list
    # append the variable array to the list, concatenate it and compute mean and std
    l = []
    total_data_mb = 0.0
    total_mask_mb = 0.0
    rss_before_loop = _rss_mb_linux()
    print(
        f"[LOOP_MEM] before_loop rss_now_mb={_format_mb(rss_before_loop)} "
        f"rss_peak_mb={_format_mb(_peak_rss_mb())}"
    )
    start_time = time.time()
    for counter, filename in enumerate(file_list, start=1):
        if args.n is not None and counter > args.n:
            break
        file_path = filename.strip()
        with nc.Dataset(file_path) as file_ds:
            array = file_ds[args.variable][:]
        l.append(array)

        data_mb, mask_mb = _array_memory_mb(array)
        total_data_mb += data_mb
        total_mask_mb += mask_mb
        rss_now_mb = _rss_mb_linux()
        rss_delta_mb = None if rss_before_loop is None or rss_now_mb is None else rss_now_mb - rss_before_loop
        print(
            f"[LOOP_MEM] i={counter} payload_data_mb={data_mb:.2f} payload_mask_mb={mask_mb:.2f} "
            f"cumulative_payload_mb={total_data_mb + total_mask_mb:.2f} "
            f"rss_now_mb={_format_mb(rss_now_mb)} rss_delta_mb={_format_mb(rss_delta_mb)} "
            f"rss_peak_mb={_format_mb(_peak_rss_mb())}"
        )
    end_time = time.time()
    print(f"Time taken to read and append variable arrays: {end_time - start_time} seconds")
    rss_after_loop = _rss_mb_linux()
    rss_loop_delta_mb = None if rss_before_loop is None or rss_after_loop is None else rss_after_loop - rss_before_loop
    print(
        f"[LOOP_MEM] after_loop arrays={len(l)} cumulative_data_mb={total_data_mb:.2f} "
        f"cumulative_mask_mb={total_mask_mb:.2f} rss_now_mb={_format_mb(rss_after_loop)} "
        f"rss_delta_mb={_format_mb(rss_loop_delta_mb)} rss_peak_mb={_format_mb(_peak_rss_mb())}"
    )
    
    # time concatenation and computation of mean and std
    rss_before_concat = _rss_mb_linux()
    start_time = time.time()
    l1 = ma.concatenate(l)
    end_time = time.time()
    print(f"Time taken to concatenate variable arrays: {end_time - start_time} seconds")
    rss_after_concat = _rss_mb_linux()
    rss_concat_delta_mb = None if rss_before_concat is None or rss_after_concat is None else rss_after_concat - rss_before_concat
    print(
        f"[CONCAT_MEM] rss_before_mb={_format_mb(rss_before_concat)} rss_after_mb={_format_mb(rss_after_concat)} "
        f"rss_delta_mb={_format_mb(rss_concat_delta_mb)} rss_peak_mb={_format_mb(_peak_rss_mb())}"
    )


    print(f"[INFO] Shape dopo concatenazione: {l1.shape}")
    print(f"[INFO] Numero totale elementi: {l1.size}")
    # time this part
    rss_before_avgstd = _rss_mb_linux()
    _log_memory("AVGSTD_MEM", rss_before_avgstd, "before_numpy_reduce")
    start_time = time.time()
    avg = np.average(l1)
    std = np.std(l1)
    end_time = time.time()
    print(f"Time taken to compute average and std: {end_time - start_time} seconds")
    _log_memory("AVGSTD_MEM", rss_before_avgstd, "after_numpy_reduce")
    print("avg", avg)
    print("std", std)


    # ok, now load the first file again

    # extract a number of files correspondig to args.n
    # put them in files list
    rss_before_files = _rss_mb_linux()
    _log_memory("FILES_MEM", rss_before_files, "before_build")
    files = []
    with open(args.file_path, 'r') as f:
        for i, line in enumerate(f):
            if args.n is not None and i >= args.n:
                break
            files.append(line.strip())
    _log_memory("FILES_MEM", rss_before_files, f"after_build files={len(files)}")
    mus = []
    vars_ = []

    rss_before_xarray_loop = _rss_mb_linux()
    _log_memory("XARRAY_LOOP_MEM", rss_before_xarray_loop, "before_loop")
    start_time = time.time()
    for counter, dataset_path in enumerate(files, start=1):
        with xr.open_dataset(dataset_path) as dataset:
            x = dataset[args.variable]
            mus.append(x.mean().item())
            vars_.append(x.var().item())
        _log_memory(
            "XARRAY_LOOP_MEM",
            rss_before_xarray_loop,
            f"i={counter} mus_len={len(mus)} vars_len={len(vars_)}"
        )
    end_time = time.time()
    print(f"Time taken to compute file-wise mean and var with xarray: {end_time - start_time} seconds")
    _log_memory("XARRAY_LOOP_MEM", rss_before_xarray_loop, f"after_loop files={len(files)}")

    rss_before_xarray_reduce = _rss_mb_linux()
    _log_memory("XARRAY_REDUCE_MEM", rss_before_xarray_reduce, "before_reduce")
    start_time = time.time()

    mu_tot = np.mean(mus)

    sigma = np.sqrt(np.mean([
        v + (m - mu_tot)**2
        for v, m in zip(vars_, mus)
    ]))
    end_time = time.time()
    print(f"Time taken to compute average and std with xarray: {end_time - start_time} seconds")
    _log_memory("XARRAY_REDUCE_MEM", rss_before_xarray_reduce, "after_reduce")
    print("avg with xarray", mu_tot)
    print("std with xarray", sigma)
