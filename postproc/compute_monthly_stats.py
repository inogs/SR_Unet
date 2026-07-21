import os

from pp_library import (
    MONTH_IDS,
)

from pp_library_nc import (
    write_stats,
    compute_list_stats
)

from pp_library_conf import read_compute_monthly_stats_conf


conf_path = os.path.join(os.path.dirname(__file__), "conf.files.dir", "conf_compute_monthly_stats.json")
conf = read_compute_monthly_stats_conf(conf_path)

path_input = conf.input_path
path_output = conf.output_path
output_folder_name = conf.output_folder_name
variable = conf.variable
jobs = conf.jobs

# get a file list of files in the input folder
# check that for every month there is a corresponding file in the input folder
# file name is month_id.txt, where month_id is the month number (01, 02, ..., 12)
list_input_files = os.listdir(path_input)
for month, month_id in MONTH_IDS.items():
    expected_file_name = f"{month_id}.txt"
    if expected_file_name not in list_input_files:
        raise FileNotFoundError(f"Expected file {expected_file_name} for month {month} not found in {path_input}")
    else:
        print(f"Found expected file {expected_file_name} for month {month} in {path_input}")


# get file path of january
january_file_path = os.path.join(path_input, f"{MONTH_IDS['jan']}.txt")
print(f"January file path: {january_file_path}")

mu_january, sigma_january = compute_list_stats(january_file_path, variable, jobs)

# use write_stats
# create folder inside output path, get folder name from input path, and write stats to that folder
# name has to be month id + month name, get month name from MONTH_IDS
output_folder_path = os.path.join(path_output, output_folder_name)
os.makedirs(output_folder_path, exist_ok=True)
# write_stats(output_folder_path, f"{MONTH_IDS['jan']}.jan", mu_january, sigma_january)

# ok, now, for every month (january included), compute the stats and write them to the output folder
# define two lists, one for mus and one for sigmas, 
mus = []
sigmas = []
for month, month_id in MONTH_IDS.items():
    month_file_path = os.path.join(path_input, f"{month_id}.txt")
    mu, sigma = compute_list_stats(month_file_path, variable, jobs)
    write_stats(output_folder_path, f"{month_id}.{month}", mu, sigma)
    mus.append(mu)
    sigmas.append(sigma)

# print another file with thre columns, month_id, mu, sigma
output_file_path = os.path.join(output_folder_path, "monthly_stats.txt")
with open(output_file_path, "w") as f:
    f.write("month_id\tmu\tsigma\n")
    for month, month_id in MONTH_IDS.items():
        mu = mus[list(MONTH_IDS.keys()).index(month)]
        sigma = sigmas[list(MONTH_IDS.keys()).index(month)]
        f.write(f"{month_id}\t{mu:.16e}\t{sigma:.16e}\n")
