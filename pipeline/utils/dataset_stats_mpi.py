from mpi4py import MPI
import argparse
import json
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import os
from types import SimpleNamespace

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = comm.Get_size()


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Convert a netCDF variable using MPI from a config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf_data_info.json"),
        help="Path to configuration file.",
    )
    return parser.parse_args()


def validate_conf_file_path(conf_path):
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")
    if not os.path.isfile(conf_path):
        raise ValueError(f"Configuration path is not a file: {conf_path}")

    if rank == 0:
        print("Configuration path check: passed")
        print(f"    Configuration path: {conf_path}")


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def validate_conf(conf):
    required_fields = [
        "folder_path",
        "output_path",
        "conversion_type",
        "variable_name",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    for field_name in required_fields:
        field_value = getattr(conf, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"Configuration field must be a non-empty string: {field_name}"
            )

    if not os.path.exists(conf.folder_path):
        raise FileNotFoundError(f"Input folder not found: {conf.folder_path}")
    if not os.path.isdir(conf.folder_path):
        raise ValueError(f"Input path is not a folder: {conf.folder_path}")
    if os.path.exists(conf.output_path) and not os.path.isdir(conf.output_path):
        raise ValueError(f"Output path is not a folder: {conf.output_path}")

    # check conversion typ, it has to be either "log" or "exp", if not raise an error
    if conf.conversion_type not in ["log", "exp"]:
        raise ValueError(
            "Configuration field conversion_type must be one of: exp, log"
        )

    if rank == 0:
        print("Configuration file check: passed")
        for field_name, field_value in sorted(vars(conf).items()):
            print(f"    {field_name}: {field_value}")

def count_valid_between_powers_of_10_from_path(path, var_key):
    ds_hf_nc = nc.Dataset(path)
    chla = ds_hf_nc.variables[var_key][:, :, :]
    valid_chla_all = chla.compressed()
    
    powers_of_10 = [10**(-n) for n in range(1, 38)]
    counts_between = {}
    for i in range(len(powers_of_10) - 1):
        lower_bound = powers_of_10[i]
        upper_bound = powers_of_10[i + 1]
        count = np.count_nonzero((valid_chla_all >= upper_bound) & (valid_chla_all < lower_bound))
        counts_between[upper_bound] = count
    # add a zero key for the count of valid Chla values equal to 0
    count_zeros = np.count_nonzero(valid_chla_all == 0)
    counts_between[0] = count_zeros
    # print this count
    # print(f"Process {rank} count of valid Chla values equal to 0 for file {path}: {count_zeros}")
    # add count of negative values, if any
    count_negative = np.count_nonzero(valid_chla_all < 0)
    counts_between['negative'] = count_negative
    ds_hf_nc.close() # close the dataset after processing
    return counts_between

def extract_zero_lowest_negative_counts(counts_dict):
    zero_count = counts_dict.get(0, 0)
    negative_count = counts_dict.get('negative', 0)

    lowest_power_of_10 = None
    for key, count in counts_dict.items():
        if count == 0 or key in (0, 'negative'):
            continue

        if isinstance(key, (int, float)) and key > 0:
            if lowest_power_of_10 is None or key < lowest_power_of_10:
                lowest_power_of_10 = key

    return zero_count, lowest_power_of_10, negative_count


def format_power_of_10(power_of_10):
    if power_of_10 is None:
        return "NA"
    return f"{power_of_10:.0e}"


def write_dataset_stats(output_path, variable_name, stats_rows):
    output_file = os.path.join(output_path, f"{variable_name}.dataset_stats.txt")

    with open(output_file, "w") as f:
        f.write(
            f"{'file_name':<24}"
            f"{'zero_count':>18}"
            f"{'min_pw_of_10':>18}"
            f"{'neg_count':>18}\n"
        )
        for _, file_name, zero_count, lowest_power_of_10, negative_count in stats_rows:
            f.write(
                f"{file_name:<24}"
                f"{zero_count:>18}"
                f"{format_power_of_10(lowest_power_of_10):>18}"
                f"{negative_count:>18}\n"
            )

    return output_file


def get_netcdf_file_list(input_dir):
    file_list = []
    for file in os.listdir(input_dir):
        if file.endswith(".nc"):
            file_list.append(os.path.join(input_dir, file))
    return sorted(file_list)


def validate_file_list(file_list, variable_name):
    if not file_list:
        raise ValueError("No netCDF files found in the input folder.")

    with nc.Dataset(file_list[0]) as dataset:
        if variable_name not in dataset.variables:
            raise ValueError(
                f"Variable {variable_name} not found in first netCDF file: {file_list[0]}"
            )


# define a function that takes a string as argument 

def get_conversion_function(conversion_type):
    if conversion_type == "log":
        return np.log
    elif conversion_type == "exp":
        return np.exp
    return None

if __name__ == "__main__":

    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    validate_conf(conf)

    folder_path = conf.folder_path
    output_path = conf.output_path
    conversion_type = conf.conversion_type
    variable_name = conf.variable_name
    conversion_function = get_conversion_function(conversion_type)

    # chek if output path exists, if not create it
    if rank == 0 and not os.path.exists(output_path):
        os.makedirs(output_path)
    comm.Barrier()
    
    file_list = get_netcdf_file_list(folder_path)
    validate_file_list(file_list, variable_name)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("This is the master process.")
        print(f"Found {len(file_list)} netCDF files in the folder.")

    assigned_data_files = list(enumerate(file_list))[rank::n_processes]
    print(f"Process {rank} processing {len(assigned_data_files)} files")

    local_stats = []
    for file_index, file in assigned_data_files:
        print(f"Process {rank} assigned file: {file}")
        counts_between = count_valid_between_powers_of_10_from_path(path= file, var_key = variable_name)
        zero_count, lowest_power_of_10, negative_count = extract_zero_lowest_negative_counts(counts_between)

        display_file_name = os.path.basename(file)
        local_stats.append(
            (
                file_index,
                display_file_name,
                zero_count,
                lowest_power_of_10,
                negative_count,
            )
        )
        del counts_between

    gathered_stats = comm.gather(local_stats, root=0)
    if rank == 0:
        stats_rows = [row for process_rows in gathered_stats for row in process_rows]
        stats_rows.sort(key=lambda row: row[0])
        output_file = write_dataset_stats(output_path, variable_name, stats_rows)
        print(f"Saved dataset statistics to {output_file}")


# end of main
# end of file
