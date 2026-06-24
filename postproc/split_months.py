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

def determine_month(month_number):
    t = int(month_number)
    if 0 <= t < 6:
        return "jan"
    elif 6 <= t < 12:
        return "feb"
    elif 12 <= t < 18:
        return "mar"
    elif 18 <= t < 24:
        return "apr"
    elif 24 <= t < 30:
        return "may"
    elif 30 <= t < 36:
        return "jun"
    elif 36 <= t < 42:
        return "jul"
    elif 42 <= t < 48:
        return "aug"
    elif 48 <= t < 54:
        return "sep"
    elif 54 <= t < 60:
        return "oct"
    elif 60 <= t < 66:
        return "nov"
    elif 66 <= t <= 72:
        return "dec"
    else:
        return None


path_output = os.path.join(os.path.dirname(__file__), "out.months")
path_input = os.path.join(os.path.dirname(__file__), "Chla.test.txt")
print(path_output)
print(path_input)


list_input_files = []
# define a dictionary with keys months of the year and values empty lists
dist_output_month_files = {
    "jan": [],
    "feb": [],
    "mar": [],
    "apr": [],
    "may": [],
    "jun": [],
    "jul": [],
    "aug": [],
    "sep": [],
    "oct": [],
    "nov": [],
    "dec": []
}

# read from input
with open(path_input, "r") as f:
    for line in f:
        list_input_files.append(line.strip())

# print first 5 items, line by line
print("First 5 items in the input file:")
for item in list_input_files[:5]:
    print(item)

for file_name in list_input_files:
    year, month = extract_year_and_season(file_name)
    # list_years.append(year)
    # list_seasons.append(month)
    season_name = determine_season(month)
    print(f"File: {file_name}, Year: {year}, Month: {month}, Season: {season_name}")
    # add file name to the corresponding month list in the dictionary
    month_name = determine_month(month)
    if month_name:
        dist_output_month_files[month_name].append(file_name)

print("Number of files in each month:")
for month, files in dist_output_month_files.items():
    print(f"{month}: {len(files)}")

# assign ids 01 to 12 to the months of the year
month_ids = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12"
}

# write 12 output files, one for each month
# create a folder
# folder name: Chla.test.months
# output file name convention: month_id +".txt"
path_output_folder = os.path.join(path_output, "Chla.test.months")
os.makedirs(path_output_folder, exist_ok=True)
for month, files in dist_output_month_files.items():
    month_id = month_ids[month]
    output_file_name = f"{month_id}.txt"
    output_file_path = os.path.join(path_output_folder, output_file_name)
    with open(output_file_path, "w") as f:
        for item in files:
            f.write("%s\n" % item)
