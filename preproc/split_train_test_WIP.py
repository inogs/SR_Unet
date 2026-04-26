import os
from typing import List
import random
import argparse
import re



def extract_year_and_season(s):
    groups = re.findall(r"\d+", s)
    if len(groups) < 2:
        return (None, None)
    return groups[-2], groups[-1]


def file_sort_key(file_name: str):
    year, season = extract_year_and_season(file_name)
    return int(year), int(season), file_name


def get_file_list(source_dir:str, test_size:float, val_size:float) -> (List[str], List[str],List[str]):
    
    seed = 42
    random.seed(seed)

    file_list = sorted(
        [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) if extract_year_and_season(f) != (None, None)],
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
        # print(f"Season: {ssn}, Remaining files: {len(remaining_files)}, Val files to select: {num_files}")
        random_files = random.sample(remaining_files, num_files)
        val_file_list += random_files

    train_file_list = [f for ssn in seasons for f in seasons[ssn] if f not in test_file_list if f not in val_file_list]
    
    # print(test_file_list)

    return test_file_list, train_file_list, val_file_list


def parse_input_parameters():

    parser = argparse.ArgumentParser(
        description="Split the dataset into train and test files."
    )
    parser.add_argument("-dp", "--data-path", required=True,
                        help="Input dataset directory.")
    parser.add_argument("-ts", "--test-size", required=True,
                        type=lambda value: float(value) if 0.0 <= float(value) <= 1.0 else (_ for _ in ()).throw(argparse.ArgumentTypeError("Value must be between 0.0 and 1.0")),
                        help="Fraction of data to reserve for test set, e.g. 0.2.")
    parser.add_argument("-vs", "--validation-size", default=0.0,
                        type=lambda value: float(value) if 0.0 <= float(value) <= 1.0 else (_ for _ in ()).throw(argparse.ArgumentTypeError("Value must be between 0.0 and 1.0")),
                        help="Fraction of training data to use as validation, e.g. 0.1.")
    args = parser.parse_args()

    return args

def print_file_list(file_list:List[str], output_file_path:str ,output_file:str) -> None:
    
    # make sure input list is not empty
    if len(file_list) != 0:
        with open(os.path.join(output_file_path, output_file), 'w') as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")
    else:
        print("Empty list")

if __name__ == '__main__':
   
    args = parse_input_parameters()

    print(f"Data path: {args.data_path}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.validation_size}")

    # check over sum of test and validation size
    if args.test_size + args.validation_size >= 1.0:
        raise ValueError("Test size and validation size must sum to less than 1.0")

    print(f"Train size: {1.0 - args.test_size - args.validation_size}")

    normalized_data_path = os.path.normpath(args.data_path)
    test_files_list, train_files_list, val_files_list = get_file_list(normalized_data_path, args.test_size, args.validation_size)

    folder_name = os.path.basename(normalized_data_path)
    train_file_name = f"{folder_name}.train.txt"
    val_file_name = f"{folder_name}.val.txt"
    test_file_name = f"{folder_name}.test.txt"

    print(f"[printing file lists]")
    parent_folder_path = os.path.dirname(normalized_data_path)
    print_file_list(train_files_list, output_file_path=parent_folder_path, output_file=train_file_name)
    print_file_list(val_files_list, output_file_path=parent_folder_path, output_file=val_file_name)
    print_file_list(test_files_list, output_file_path=parent_folder_path, output_file=test_file_name)
    print(f"[ending execution]")
