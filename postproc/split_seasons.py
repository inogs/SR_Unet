import argparse
import os

from pp_library import (
    extract_year_and_period_id,
    determine_season,
    make_season_file_dict,
    read_file_list,
)
from pp_library_conf import read_split_seasons_conf


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Split a file list into per-season file lists."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf.files.dir", "conf_split_seasons.json"),
        help="Path to configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_input_parameters()
    conf_split_seasons = read_split_seasons_conf(args.config)
    print(f"Configuration for split seasons: {conf_split_seasons}")

    path_input = conf_split_seasons.input_path
    path_output = conf_split_seasons.output_path
    folder_name = conf_split_seasons.output_folder_name

    list_input_files = read_file_list(path_input)
    dist_output_season_files = make_season_file_dict()

    for file_name in list_input_files:
        year, period_id = extract_year_and_period_id(file_name)
        season_name = determine_season(period_id)
        print(f"File: {file_name}, Year: {year}, Period ID: {period_id}, Season: {season_name}")
        if season_name:
            dist_output_season_files[season_name].append(file_name)

    print("Number of files in each season:")
    for season, files in dist_output_season_files.items():
        print(f"{season}: {len(files)}")

    path_output_folder = os.path.join(path_output, folder_name)
    os.makedirs(path_output_folder, exist_ok=True)
    for season, files in dist_output_season_files.items():
        output_file_path = os.path.join(path_output_folder, f"{season}.txt")
        with open(output_file_path, "w") as f:
            for item in files:
                f.write(f"{item}\n")
