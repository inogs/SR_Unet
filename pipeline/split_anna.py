import os
from typing import List
import random
import argparse
import re
import json
from types import SimpleNamespace

def extract_year_and_season(s):
    file_name = os.path.basename(s)
    groups = re.findall(r"\d+", file_name)
    if len(groups) < 2:
        return (None, None)
    return groups[-2], groups[-1]


def file_sort_key(file_name: str):
    year, season = extract_year_and_season(file_name)
    return int(year), int(season), file_name


def get_file_list_random(source_dir:str, test_size:float, val_size:float, seed:int) -> (List[str], List[str],List[str]):
    random.seed(seed)

    source_dir = os.path.abspath(source_dir)

    file_list = sorted(
        [os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) if extract_year_and_season(f) != (None, None)],
        key=file_sort_key,
    )

    seasons = {"winter" : [], "spring" : [], "summer" : [], "autumn" : []}
    for file_name in file_list:
        # get second argument of extract_year_and_season, which is the season number (0-72), cast it to int
        t = int(extract_year_and_season(file_name)[1])    
        if 0 <= t < 18:
            seasons["winter"].append(file_name)
        elif 18 <= t < 36:
            seasons["spring"].append(file_name)
        elif 36 <= t < 54:
            seasons["summer"].append(file_name)
        elif 54 <= t <= 72:
            seasons["autumn"].append(file_name)

    test_file_list = []
    for ssn in seasons:
        l = seasons[ssn]
        num_files = min(len(l), max(1, int(len(l) * test_size)))
        random_files = random.sample(l, num_files)
        test_file_list += random_files

    val_file_list = []
    for ssn in seasons:
        l = seasons[ssn]
        remaining_files = [f for f in l if f not in test_file_list]
        num_files = min(len(remaining_files), max(1, int(len(remaining_files) * val_size)))
        random_files = random.sample(remaining_files, num_files)
        val_file_list += random_files

    train_file_list = [f for ssn in seasons for f in seasons[ssn] if f not in test_file_list if f not in val_file_list]
    
    # print(test_file_list)

    return test_file_list, train_file_list, val_file_list


def get_file_list_temporal_split(source_dir: str, train_years: int) -> tuple[List[str], List[str]]:

    source_dir = os.path.abspath(source_dir)

    # Extract year and temporal index from filename
    def extract_time(file_name):
        name = os.path.basename(file_name)
        numbers = re.findall(r"\d{4}|\d{3}", name) # collect all int and positive number with 4cifre and 3cifre

        if len(numbers) < 2:
            return None, None

        year = int(numbers[0])
        timestep = int(numbers[1])

        return year, timestep


    # collect valid files
    file_list = []
    ignored_files = 0

    for f in os.listdir(source_dir):

        full_path = os.path.join(source_dir, f)

        if not os.path.isfile(full_path):
            continue

        year, timestep = extract_time(f)

        if year is None:
            ignored_files += 1
        else:
            file_list.append(full_path)

    if ignored_files > 0:
        print(
            f"WARNING: {ignored_files} file(s) were ignored because their filename "
            "does not contain a valid year and timestep."
    )


    if len(file_list) == 0:
        raise ValueError(
            f"No files with recognizable temporal information found in {source_dir}" # if ALL files don't have recognizable temporal info
        )


    # temporal ordering
    file_list = sorted(
        file_list,
        key=lambda x: (
            extract_time(x)[0],
            extract_time(x)[1],
            x
        )
    )


    # identify available years
    years = sorted(
        list(
            set(
                extract_time(f)[0]
                for f in file_list
            )
        )
    )


    if train_years > len(years):
        raise ValueError(
            f"train_years={train_years} but dataset contains only {len(years)} years"
        )


    print("YEARS =", years)
    print("len(years) =", len(years))
    print("train_years =", train_years)

    train_year_list = years[:train_years]
    test_year_list = years[train_years:]

    print("TRAIN YEARS =", train_year_list)
    print("TEST YEARS =", test_year_list)


    train_files = [
        f for f in file_list
        if extract_time(f)[0] in train_year_list
    ]


    test_files = [
        f for f in file_list
        if extract_time(f)[0] in test_year_list
    ]

    return train_files, test_files


def parse_input_parameters():

    parser = argparse.ArgumentParser(
        description="Split the dataset into train, validation and test files from a configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default='/leonardo/home/userexternal/adelsavi/git/codice_fede/pipeline/conf.files.dir/conf_split_anna.json',
        help="Path to split configuration file.",
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
    with open(conf_path, 'r') as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def validate_conf(conf) -> None:

    required_fields = [
        "data_path",
        "output_path",
        "label",
        "split_method",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(
                f"Missing configuration field: {field_name}"
            )


    if not os.path.exists(conf.data_path):
        raise ValueError(
            f"Data path does not exist: {conf.data_path}"
        )

    # if not os.path.exists(conf.output_path):
    #     raise ValueError(
    #         f"Output path does not exist: {conf.output_path}"
    #     )


    if conf.split_method not in ["random", "temporal"]:
        raise ValueError(
            "split_method must be 'random' or 'temporal'"
        )


    if conf.split_method == "random":

        required_random = [
            "test_size",
            "validation_size",
            "seed",
        ]

        for field_name in required_random:
            if not hasattr(conf, field_name):
                raise AttributeError(
                    f"Missing random split parameter: {field_name}"
                )


    elif conf.split_method == "temporal":

        if not hasattr(conf, "train_years"):
            raise AttributeError(
                "Temporal split requires train_years"
            )


    if not isinstance(conf.label, str) or not conf.label.strip():
        raise ValueError(
            "Label must be a non-empty string"
        )


def print_file_list(file_list: List[str], output_file_path: str, output_file: str) -> None:
    output_file_path = os.path.abspath(output_file_path)
    os.makedirs(output_file_path, exist_ok=True)

    with open(os.path.join(output_file_path, output_file), 'w') as f:
        for file_path in file_list:
            f.write(f"{file_path}\n")

    if len(file_list) == 0:
        print(f"Empty list -> created empty file: {output_file}")

if __name__ == '__main__':
   
    args = parse_input_parameters()
    conf = read_conf_file(args.config)
    validate_conf(conf)

    output_folder_path = os.path.abspath(conf.output_path)
    os.makedirs(output_folder_path, exist_ok=True)

    if conf.split_method == "random":
        test_files_list, train_files_list, val_files_list = get_file_list_random(
            source_dir=conf.data_path,
            test_size=conf.test_size,
            val_size=conf.validation_size,
            seed=conf.seed,
        )
    elif conf.split_method == "temporal":

        train_files_list, test_files_list = get_file_list_temporal_split(
            source_dir=conf.data_path,
            train_years=conf.train_years,
        )

    print_file_list(train_files_list, 
        output_file_path=output_folder_path,
        output_file=f"{conf.label}.train.txt")
    
    if conf.split_method == "random":
        print_file_list(val_files_list,  
            output_file_path=output_folder_path,
            output_file=f"{conf.label}.val.txt")
        
    print_file_list(test_files_list,  
        output_file_path=output_folder_path,
        output_file=f"{conf.label}.test.txt"
)
    





