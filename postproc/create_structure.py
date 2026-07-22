import argparse
import json
import os

from pp_library_conf import read_create_structure_conf


DATA_TYPE_SPECS = [
    ("original", "variables", False),
    ("converted", "conversion_variables", True),
    ("treshold", "treshold_variables", False),
    ("treshold.converted", "treshold_variables", True),
]

ANALYTICS_GROUPS = ["input", "target", "predictions"]
ANALYTICS_PERIODS = ["montly", "seasonal"]

PREPROC_STRUCTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pipeline", "conf.files.dir", "preproc_structure.json"
)
TRAINING_STRUCTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "training", "conf", "training_structure.json"
)


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Create the postproc directory structure from a JSON configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf.files.dir", "conf_orchestrator.json"),
        help="Path to configuration file.",
    )
    return parser.parse_args()


def read_tree_file(path):
    with open(path) as f:
        return json.load(f)["tree"]


def check_tree(root_path, tree):
    for folder_name in tree:
        folder_path = os.path.join(root_path, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Expected folder not found: {folder_path}")
    print(f"Structure check passed: {root_path}")


def build_pair_name(var_input, var_target, conversion_type, suffixed):
    if suffixed:
        return f"{var_input}.{conversion_type}.{var_target}.{conversion_type}"
    return f"{var_input}.{var_target}"


def analytics_category_name(group, category):
    if group == "input" and category == "original":
        return "interpolated"
    return category


def create_postproc_structure(root_path, conf):
    staged_categories = vars(conf.mpi_operations.categories)

    for folder_name in ("conf.files", "log"):
        folder_path = os.path.join(root_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

    for category, variables_key, suffixed in DATA_TYPE_SPECS:
        variables = getattr(conf, variables_key)
        stage_conf = vars(staged_categories[category]) if category in staged_categories else None

        for var_input, var_target in vars(variables).items():
            pair_name = build_pair_name(var_input, var_target, conf.conversion_type, suffixed)

            if stage_conf:
                pred_stages = ["raw", "treshold", f"treshold.{stage_conf['conversion_type']}"]
            else:
                pred_stages = [None]

            for stage in pred_stages:
                pred_dir = os.path.join(root_path, "predictions", "test.dataset", category, *filter(None, [stage]), pair_name)
                os.makedirs(pred_dir, exist_ok=True)
                print(f"Created folder: {pred_dir}")

            for group in ANALYTICS_GROUPS:
                analytics_category = analytics_category_name(group, category)

                if group == "predictions" and stage_conf:
                    analytics_stages = ["raw", f"treshold.{stage_conf['conversion_type']}"]
                else:
                    analytics_stages = [None]

                for stage in analytics_stages:
                    for period in ANALYTICS_PERIODS:
                        for tree_name in ("analytics", "file.list"):
                            tree_dir = os.path.join(
                                root_path, tree_name, "test.dataset", group, analytics_category,
                                *filter(None, [stage]), period, pair_name
                            )
                            os.makedirs(tree_dir, exist_ok=True)
                            print(f"Created folder: {tree_dir}")


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_create_structure_conf(args.config)

    check_tree(conf.path_preproc_dir, read_tree_file(PREPROC_STRUCTURE_PATH))
    check_tree(conf.path_training_dir, read_tree_file(TRAINING_STRUCTURE_PATH))

    create_postproc_structure(conf.path_postproc_dir, conf)
