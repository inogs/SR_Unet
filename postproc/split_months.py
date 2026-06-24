import os

from pp_library import (
    extract_year_and_period_id,
    determine_season,
    determine_month,
    MONTH_IDS,
    make_month_file_dict,
    read_file_list
)

from pp_library_conf import ( read_split_months_conf)

# setup block
conf_split_months_path = os.path.join(os.path.dirname(__file__), "conf_split_months.json")
conf_split_months = read_split_months_conf(conf_split_months_path)
print(f"Configuration for split months: {conf_split_months}")
path_input = conf_split_months.input_path
path_output = conf_split_months.output_path
folder_name = conf_split_months.output_folder_name

# initial structures
list_input_files = read_file_list(path_input)
dist_output_month_files = make_month_file_dict()

# get months
for file_name in list_input_files:
    year, period_id = extract_year_and_period_id(file_name)
    season_name = determine_season(period_id)
    print(f"File: {file_name}, Year: {year}, Period ID: {period_id}, Season: {season_name}")
    month_name = determine_month(period_id)
    if month_name:
        dist_output_month_files[month_name].append(file_name)

print("Number of files in each month:")
for month, files in dist_output_month_files.items():
    print(f"{month}: {len(files)}")

# print outputs to file
path_output_folder = os.path.join(path_output, folder_name)
os.makedirs(path_output_folder, exist_ok=True)
for month, files in dist_output_month_files.items():
    month_id = MONTH_IDS[month]
    output_file_name = f"{month_id}.txt"
    output_file_path = os.path.join(path_output_folder, output_file_name)
    with open(output_file_path, "w") as f:
        for item in files:
            f.write("%s\n" % item)
