import argparse
import json
import os


# (data_type, variables_key, suffix_both_with_conversion_type, pt_file_prefix)
DATA_TYPE_SPECS = [
    ("original", "variables", False, ""),
    ("converted", "conversion_variables", True, ""),
    ("treshold", "treshold_variables", False, "treshold."),
    ("treshold.converted", "treshold_variables", True, "treshold."),
]

PT_TYPES = ["train", "val", "test"]

JOB_TEMPLATE = """#!/bin/bash
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
##SBATCH 30 min is the max for debug queue
#SBATCH --time={time}           # walltime (HH:MM:SS)
#SBATCH --account={account} # project/account
#SBATCH --partition={partition} # partition/queue
#SBATCH -J {job_name}     # job name
#SBATCH -o {out_dir}/%x-%j.log         # stdout file (%x=jobname, %j=jobid)
#SBATCH -e {out_dir}/%x-%j.log         # stderr file (%x=jobname, %j=jobid)

## DEBUG ON
{debug_qos_line}

echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total --format=csv

eval "$({conda_base_path}/bin/conda shell.bash hook)"
conda activate {conda_env}

srun --mpi=pmix_v3 python {train_script_path} \\
-c {conf_json_path}
"""


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Create the training directory structure from a JSON configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(
            os.path.dirname(__file__), "conf", "conf_training_orchestrator.json"
        ),
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f)


def build_pair_name(var_input, var_target, conversion_type, suffixed):
    if suffixed:
        return f"{var_input}.{conversion_type}.{var_target}.{conversion_type}"
    return f"{var_input}.{var_target}"


def build_train_conf(train_template, path_pt_dir, pt_basename, out_dir):
    train_conf = dict(train_template)
    train_conf["output_path"] = out_dir

    for pt_type in PT_TYPES:
        train_conf[f"var_{pt_type}_path"] = os.path.join(
            path_pt_dir, f"{pt_basename}.{pt_type}.dataset.pt"
        )
        train_conf[f"river_{pt_type}_path"] = os.path.join(
            path_pt_dir, f"rivers_{pt_type}.pt"
        )

    return train_conf


def build_debug_qos_line(debug_conf):
    line = f"#SBATCH --qos={debug_conf['qos']}"
    if debug_conf["enabled"]:
        return line
    return f"# {line}"


def build_job_script(job_conf, out_dir, conf_json_path):
    train_script_path = os.path.join(job_conf["path_source_dir"], "train_prod.py")

    return JOB_TEMPLATE.format(
        nodes=job_conf["nodes"],
        ntasks_per_node=job_conf["ntasks_per_node"],
        gres=job_conf["gres"],
        cpus_per_task=job_conf["cpus_per_task"],
        time=job_conf["time"],
        account=job_conf["account"],
        partition=job_conf["partition"],
        job_name=job_conf["job_name"],
        out_dir=out_dir,
        debug_qos_line=build_debug_qos_line(job_conf["debug"]),
        conda_base_path=job_conf["conda_base_path"],
        conda_env=job_conf["conda_env"],
        train_script_path=train_script_path,
        conf_json_path=conf_json_path,
    )


def create_training_structure(path_training_dir, general_conf, train_template, job_conf):
    path_pt_dir = os.path.join(
        general_conf["path_preproc_dir"], "splits", general_conf["split_label"], "pt.files"
    )

    os.makedirs(path_training_dir, exist_ok=True)
    print(f"Created folder: {path_training_dir}")

    for data_type, variables_key, suffixed, pt_prefix in DATA_TYPE_SPECS:
        data_type_dir = os.path.join(path_training_dir, data_type)
        os.makedirs(data_type_dir, exist_ok=True)
        print(f"Created folder: {data_type_dir}")

        for var_input, var_target in general_conf[variables_key].items():
            pair_name = build_pair_name(
                var_input, var_target, general_conf["conversion_type"], suffixed
            )
            pt_basename = f"{pt_prefix}{pair_name}"

            pair_dir = os.path.join(data_type_dir, pair_name)
            os.makedirs(pair_dir, exist_ok=True)
            print(f"Created folder: {pair_dir}")

            out_dir = os.path.join(pair_dir, "out")
            os.makedirs(out_dir, exist_ok=True)
            print(f"Created folder: {out_dir}")

            train_conf = build_train_conf(train_template, path_pt_dir, pt_basename, out_dir)
            conf_path = os.path.join(pair_dir, "conf.json")
            with open(conf_path, "w") as f:
                json.dump(train_conf, f, indent=4)
                f.write("\n")
            print(f"Created file: {conf_path}")

            job_script = build_job_script(job_conf, out_dir, conf_path)
            job_path = os.path.join(pair_dir, "job.sh")
            with open(job_path, "w") as f:
                f.write(job_script)
            print(f"Created file: {job_path}")


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_conf_file(args.config)
    general_conf = conf["general"]
    train_template = conf["train_conf"]
    job_conf = conf["job"]

    print(f"path_preproc_dir: {general_conf['path_preproc_dir']}")
    if not os.path.exists(general_conf["path_preproc_dir"]):
        raise FileNotFoundError(f"Preproc directory not found: {general_conf['path_preproc_dir']}")

    print(f"path_source_dir: {job_conf['path_source_dir']}")
    if not os.path.exists(job_conf["path_source_dir"]):
        raise FileNotFoundError(f"Source directory not found: {job_conf['path_source_dir']}")

    print(f"path_training_dir: {general_conf['path_training_dir']}")

    create_training_structure(general_conf["path_training_dir"], general_conf, train_template, job_conf)
