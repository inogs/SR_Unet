import argparse
import json
import os


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Create a directory tree from a JSON configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "conf.files.dir",
            "conf_create_structure_v3.json",
        ),
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f)


def create_tree(root_path, tree):
    os.makedirs(root_path, exist_ok=True)
    print(f"Created folder: {root_path}")

    for folder_name, subtree in tree.items():
        folder_path = os.path.join(root_path, folder_name)
        create_tree(folder_path, subtree)


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_conf_file(args.config)

    create_tree(conf["root"], conf["tree"])

