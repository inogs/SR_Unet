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


def get_file_list(source_dir:str, test_size:float, val_size:float, seed:int) -> (List[str], List[str],List[str]):
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


def parse_input_parameters():

    parser = argparse.ArgumentParser(
        description="Split the dataset into train, validation and test files from a configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf_split.json"),
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
        "test_size",
        "validation_size",
        "seed",
        "label",
    ]

    for field_name in required_fields:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {field_name}")

    if not os.path.exists(conf.data_path):
        raise ValueError(f"Data path does not exist: {conf.data_path}")

    if not os.path.exists(conf.output_path):
        raise ValueError(f"Output path does not exist: {conf.output_path}")

    if not isinstance(conf.label, str) or not conf.label.strip():
        raise ValueError(f"Label must be a non-empty string: {conf.label}")

    if not 0.0 <= float(conf.test_size) <= 1.0:
        raise ValueError(f"Test size must be between 0.0 and 1.0: {conf.test_size}")

    if not 0.0 <= float(conf.validation_size) <= 1.0:
        raise ValueError(f"Validation size must be between 0.0 and 1.0: {conf.validation_size}")

    print(f"Data path: {conf.data_path}")
    print(f"Output path: {conf.output_path}")
    print(f"Test size: {conf.test_size}")
    print(f"Validation size: {conf.validation_size}")
    print(f"Seed: {conf.seed}")
    print(f"Label: {conf.label}")

    if conf.test_size + conf.validation_size >= 1.0:
        raise ValueError("Test size and validation size must sum to less than 1.0")

    print(f"Train size: {1.0 - conf.test_size - conf.validation_size}")


def print_file_list(file_list:List[str], output_file_path:str ,output_file:str) -> None:
    if len(file_list) != 0:
        with open(os.path.join(output_file_path, output_file), 'w') as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")
    else:
        print("Empty list")

if __name__ == '__main__':
   
    args = parse_input_parameters()
    conf = read_conf_file(args.config)
    validate_conf(conf)

    output_folder_path = os.path.abspath(conf.output_path)
    os.makedirs(output_folder_path, exist_ok=True)

    test_files_list, train_files_list, val_files_list = get_file_list(
        source_dir=conf.data_path,
        test_size=conf.test_size,
        val_size=conf.validation_size,
        seed=conf.seed,
    )

    print_file_list(train_files_list, 
        output_file_path=output_folder_path,
        output_file=f"{conf.label}.train.txt")
    print_file_list(val_files_list,  
        output_file_path=output_folder_path,
        output_file=f"{conf.label}.val.txt")
    print_file_list(test_files_list,  
        output_file_path=output_folder_path,
        output_file=f"{conf.label}.test.txt"
)
    
