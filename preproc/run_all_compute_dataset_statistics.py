#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys


# Edit these values directly in the script.
TXT_ROOT = Path("/leonardo_scratch/large/userexternal/gzuccari/iCMS_nc/iCMS_nc.split.07.02.01.seed.42")
OUTPUT_ROOT = Path("/leonardo_scratch/large/userexternal/gzuccari/iCMS_nc/iCMS_nc.stats.07.02.01.seed.42")
COMPUTE_SCRIPT = Path("/leonardo/home/userexternal/gzuccari/git/OGS/SR_Unet/preproc/compute_dataset_statistics.py")
MAX_FILES = None
JOBS = 12
RECURSIVE = False


def find_txt_files(root: Path):
    pattern = "**/*.txt" if RECURSIVE else "*.txt"
    txt_files = []

    for txt_path in sorted(root.glob(pattern)):
        if not txt_path.is_file():
            continue
        if txt_path.name == "log.txt":
            continue
        if txt_path.name.startswith("stat."):
            continue

        first_line = first_non_empty_line(txt_path)
        if first_line is None:
            continue
        if not first_line.endswith(".nc"):
            continue

        txt_files.append(txt_path)

    return txt_files


def first_non_empty_line(txt_path: Path):
    with txt_path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def variable_from_txt_name(txt_path: Path):
    return txt_path.name.split(".")[0]


def build_command(txt_path: Path, variable: str):
    command_parts = [
        sys.executable,
        str(COMPUTE_SCRIPT),
        "-dp",
        str(txt_path),
        "-v",
        variable,
        "-j",
        str(JOBS),
        "-op",
        str(OUTPUT_ROOT),
    ]

    if MAX_FILES is not None:
        command_parts.extend(["-n", str(MAX_FILES)])

    return command_parts


def main():
    txt_files = find_txt_files(TXT_ROOT)

    if not txt_files:
        raise FileNotFoundError(f"No input txt files found in {TXT_ROOT}")

    print(f"Found {len(txt_files)} txt files in {TXT_ROOT}")

    for txt_path in txt_files:
        variable = variable_from_txt_name(txt_path)
        command = build_command(txt_path, variable)

        print(f"Launching for {txt_path} with -v {variable}")
        subprocess.run(
            command,
            check=True,
            cwd=str(COMPUTE_SCRIPT.parent),
        )


if __name__ == "__main__":
    main()
