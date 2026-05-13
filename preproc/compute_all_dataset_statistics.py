#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for all txt file lists from a config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "conf_all_stat.json",
        ),
        help="Path to configuration file.",
    )
    return parser.parse_args()


def main():
    args = parse_input_parameters()
    compute_script = os.path.join(
        os.path.dirname(__file__),
        "compute_dataset_statistics.py",
    )

    subprocess.run(
        [sys.executable, compute_script, "-c", args.config],
        check=True,
        cwd=os.path.dirname(compute_script),
    )


if __name__ == "__main__":
    main()
