import argparse
import os

from pp_library import SEASON_IDS
from pp_library_conf import read_compute_season_field_stats_conf
from pp_library_nc import compute_list_field_stats, write_field


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Compute seasonal mean/std fields from per-season file lists."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf.files.dir", "conf_compute_season_stat_field.json"),
        help="Path to configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_compute_season_field_stats_conf(args.config)

    path_input = conf.input_path
    path_output = conf.output_path
    output_folder_name = conf.output_folder_name
    variable = conf.variable
    jobs = conf.jobs

    list_input_files = os.listdir(path_input)
    for season in SEASON_IDS:
        expected_file_name = f"{season}.txt"
        if expected_file_name not in list_input_files:
            raise FileNotFoundError(f"Expected file {expected_file_name} for season {season} not found in {path_input}")

    output_folder_path = os.path.join(path_output, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    for season, season_id in SEASON_IDS.items():
        mean_path = os.path.join(output_folder_path, f"{season_id}.{season}.mean.nc")
        std_path = os.path.join(output_folder_path, f"{season_id}.{season}.std.nc")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            print(f"Output files for season {season} already exist. Skipping computation.")
            continue

        season_file_path = os.path.join(path_input, f"{season}.txt")
        mean_field, std_field, reference_file = compute_list_field_stats(season_file_path, variable, jobs)

        write_field(reference_file, mean_path, variable, mean_field)
        write_field(reference_file, std_path, variable, std_field)
