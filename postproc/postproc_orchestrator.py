import argparse
import json
import os
import shlex
from types import SimpleNamespace


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Run the postproc orchestrator."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "conf.files.dir",
            "conf_orchestrator.json",
        ),
        help="Path to the general postproc configuration file.",
    )
    return parser.parse_args()


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def validate_conf_file_path(conf_path):
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")
    if not os.path.isfile(conf_path):
        raise ValueError(f"Configuration path is not a file: {conf_path}")


def resolve_postproc_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(os.path.dirname(__file__), path_value)


def execute_command(cmd, label):
    command = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"[postproc_orchestrator] Running {label}: {command}")

    return_code = os.system(command)
    if return_code == 0:
        print(f"[postproc_orchestrator] {label} completed successfully.")
        return True

    exit_code = os.waitstatus_to_exitcode(return_code)
    print(
        f"Warning: {label} failed with return code {exit_code}. "
        "Stopping orchestrator."
    )
    return False


def build_step_command(step, config_path):
    script_path = resolve_postproc_path(step.script)
    return ["python", script_path, "--config", config_path]


def run_pipeline(conf, config_path):
    for step in conf.postproc_orchestrator.steps:
        if hasattr(step, "enabled") and not step.enabled:
            print(f"[postproc_orchestrator] Skipping disabled step: {step.name}")
            continue

        cmd = build_step_command(step, config_path)
        if not execute_command(cmd, step.name):
            return False

    return True


if __name__ == "__main__":
    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)

    print(f"[postproc_orchestrator] Configuration: {args.config}")
    print(f"[postproc_orchestrator] path_postproc_dir: {conf.path_postproc_dir}")

    success = run_pipeline(conf, args.config)
    if not success:
        raise SystemExit(1)
