import json
import os
import subprocess
from pathlib import Path


path_predictions_v2 = "/leonardo_scratch/large/userexternal/gzuccari/TEST/predictions.v2"
path_postprocessing_v2 = "/leonardo_scratch/large/userexternal/gzuccari/TEST/postprocessing.v2"

global_jobs = 32

postproc_dir = Path(__file__).resolve().parent
split_months_script = postproc_dir / "split_months.py"
compute_monthly_stats_script = postproc_dir / "compute_monthly_stats.py"
split_seasons_script = postproc_dir / "split_seasons.py"
compute_season_stat_field_script = postproc_dir / "compute_season_stat_field.py"
conf_split_months_path = postproc_dir / "conf_split_months.json"
conf_compute_monthly_stats_path = postproc_dir / "conf_compute_monthly_stats.json"
conf_split_seasons_path = postproc_dir / "conf_split_seasons.json"
conf_compute_season_stat_field_path = postproc_dir / "conf_compute_season_stat_field.json"

domains = {
    "input": {
        "chl": "chl",
        "po4": "po4",
        "no3": "no3",
    },
    "target": {
        "Chla": "Chla",
        "N1p": "N1p",
        "N3n": "N3n",
    },
    "normal.net": {
        "chl.Chla": "Chla",
        "po4.N1p": "N1p",
        "no3.N3n": "N3n",
    },
    "log.net": {
        "chl.log.Chla.log": "Chla",
        "po4.log.N1p.log": "N1p",
        "no3.log.N3n.log": "N3n",
    },
}

levels = ["original", "treshold", "treshold.log"]


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def run_script(script_path):
    subprocess.run(["python", str(script_path)], check=True, cwd=str(postproc_dir))


def ensure_layout():
    for root_name in ["montly", "seasonal"]:
        Path(path_postprocessing_v2, root_name).mkdir(parents=True, exist_ok=True)

    for section_name in ["montly.file.lists", "montly.stats"]:
        for domain_name in domains:
            for level_name in levels:
                Path(
                    path_postprocessing_v2,
                    "montly",
                    section_name,
                    domain_name,
                    level_name,
                ).mkdir(parents=True, exist_ok=True)

    for section_name in ["seasonal.file.lists", "seasonal.stat.fields"]:
        for domain_name in domains:
            for level_name in levels:
                Path(
                    path_postprocessing_v2,
                    "seasonal",
                    section_name,
                    domain_name,
                    level_name,
                ).mkdir(parents=True, exist_ok=True)


def get_prediction_file_list(domain_name, level_name, entry_name):
    file_list_path = Path(
        path_predictions_v2,
        domain_name,
        "file.lists",
        level_name,
        f"{entry_name}.txt",
    )
    if not file_list_path.is_file():
        raise FileNotFoundError(f"Prediction file list not found: {file_list_path}")
    return file_list_path


def split_months(domain_name, level_name, entry_name):
    input_file_list = get_prediction_file_list(domain_name, level_name, entry_name)
    output_path = Path(
        path_postprocessing_v2,
        "montly",
        "montly.file.lists",
        domain_name,
        level_name,
    )

    conf = {
        "input_path": str(input_file_list),
        "output_path": str(output_path),
        "output_folder_name": entry_name,
    }

    write_json(conf_split_months_path, conf)
    run_script(split_months_script)

    return output_path / entry_name


def compute_monthly_stats(domain_name, level_name, entry_name, variable_name, split_path):
    output_path = Path(
        path_postprocessing_v2,
        "montly",
        "montly.stats",
        domain_name,
        level_name,
    )

    conf = {
        "input_path": str(split_path),
        "output_path": str(output_path),
        "output_folder_name": entry_name,
        "variable": variable_name,
        "jobs": global_jobs,
    }

    write_json(conf_compute_monthly_stats_path, conf)
    run_script(compute_monthly_stats_script)


def process_monthly_stats():
    for domain_name, entries in domains.items():
        for level_name in levels:
            for entry_name, variable_name in entries.items():
                split_path = split_months(domain_name, level_name, entry_name)
                compute_monthly_stats(
                    domain_name=domain_name,
                    level_name=level_name,
                    entry_name=entry_name,
                    variable_name=variable_name,
                    split_path=split_path,
                )


def split_seasons(domain_name, level_name, entry_name):
    input_file_list = get_prediction_file_list(domain_name, level_name, entry_name)
    output_path = Path(
        path_postprocessing_v2,
        "seasonal",
        "seasonal.file.lists",
        domain_name,
        level_name,
    )

    conf = {
        "input_path": str(input_file_list),
        "output_path": str(output_path),
        "output_folder_name": entry_name,
    }

    write_json(conf_split_seasons_path, conf)
    run_script(split_seasons_script)

    return output_path / entry_name


def compute_season_stat_fields(
    domain_name, level_name, entry_name, variable_name, split_path
):
    output_path = Path(
        path_postprocessing_v2,
        "seasonal",
        "seasonal.stat.fields",
        domain_name,
        level_name,
    )

    conf = {
        "input_path": str(split_path),
        "output_path": str(output_path),
        "output_folder_name": entry_name,
        "variable": variable_name,
        "jobs": global_jobs,
    }

    write_json(conf_compute_season_stat_field_path, conf)
    run_script(compute_season_stat_field_script)


def process_seasonal_stat_fields():
    for domain_name, entries in domains.items():
        for level_name in levels:
            for entry_name, variable_name in entries.items():
                split_path = split_seasons(domain_name, level_name, entry_name)
                compute_season_stat_fields(
                    domain_name=domain_name,
                    level_name=level_name,
                    entry_name=entry_name,
                    variable_name=variable_name,
                    split_path=split_path,
                )


if __name__ == "__main__":
    if not os.path.isdir(path_predictions_v2):
        raise FileNotFoundError(f"Predictions v2 folder not found: {path_predictions_v2}")

    ensure_layout()
    process_monthly_stats()
    process_seasonal_stat_fields()
