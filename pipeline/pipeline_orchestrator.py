import argparse
import json
import os
import shlex
from types import SimpleNamespace


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline orchestrator."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "conf.files.dir",
            "conf_general.json",
        ),
        help="Path to the general pipeline configuration file.",
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


def resolve_pipeline_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(os.path.dirname(__file__), path_value)


def execute_command(cmd, label):
    command = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"[pipeline_orchestrator] Running {label}: {command}")

    return_code = os.system(command)
    if return_code == 0:
        print(f"[pipeline_orchestrator] {label} completed successfully.")
        return True

    exit_code = os.waitstatus_to_exitcode(return_code)
    print(
        f"Warning: {label} failed with return code {exit_code}. "
        "Stopping pipeline."
    )
    return False


def write_create_structure_sources_config(conf):
    sources_conf = {
        "path_input_dir": conf.path_input_dir,
        "path_target_dir": conf.path_target_dir,
        "path_rivers_dir": conf.path_rivers_dir,
    }

    sources_config_path = resolve_pipeline_path(
        os.path.join("conf.files.dir", "conf_create_structure.json")
    )
    with open(sources_config_path, "w") as f:
        json.dump(sources_conf, f, indent=4)
        f.write("\n")
    print(f"[pipeline_orchestrator] Sources configuration written: {sources_config_path}")

    return sources_config_path


def build_step_command(step, conf):
    script_path = resolve_pipeline_path(step.script)
    cmd = ["python", script_path]

    if hasattr(step, "config"):
        config_path = resolve_pipeline_path(step.config)
        cmd.extend(["--config", config_path])

    if step.name == "create_structure":
        sources_config_path = write_create_structure_sources_config(conf)
        cmd.extend(["--sources-config", sources_config_path])
        cmd.extend(["--root", conf.path_preproc_dir])

    return cmd


def check_conf(conf):
    enabled_steps = [
        step.name
        for step in conf.pipeline_orchestrator.steps
        if not hasattr(step, "enabled") or step.enabled
    ]

    if "mpi_operations" in enabled_steps and "multithread_operations" in enabled_steps:
        print(
            "Warning: mpi_operations and multithread_operations are both enabled. "
            "Enable only one execution layer. Exiting."
        )
        raise SystemExit(1)


def run_pipeline(conf):
    for step in conf.pipeline_orchestrator.steps:
        if hasattr(step, "enabled") and not step.enabled:
            print(f"[pipeline_orchestrator] Skipping disabled step: {step.name}")
            continue

        cmd = build_step_command(step, conf)
        if not execute_command(cmd, step.name):
            return False

    return True


if __name__ == "__main__":
    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    check_conf(conf)

    print(f"[pipeline_orchestrator] Configuration: {args.config}")
    print(f"[pipeline_orchestrator] path_preproc_dir: {conf.path_preproc_dir}")
    print(f"[pipeline_orchestrator] number_of_processes: {conf.mpi_operations.number_of_processes}")
    print(f"[pipeline_orchestrator] number_of_threads: {conf.multithread_operations.number_of_threads}")

    success = run_pipeline(conf)
    if not success:
        raise SystemExit(1)
