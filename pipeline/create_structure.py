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
            "preproc_structure.json",
        ),
        help="Path to the JSON file describing the directory tree.",
    )
    parser.add_argument(
        "-s",
        "--sources-config",
        default=os.path.join(
            os.path.dirname(__file__),
            "conf.files.dir",
            "conf_create_structure.json",
        ),
        help="Path to the JSON file with the source data paths to link "
        "(path_input_dir, path_target_dir, path_rivers_dir).",
    )
    parser.add_argument(
        "-r",
        "--root",
        required=True,
        help="Root path where the directory tree is created.",
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


def make_symlink(source_path, link_path):
    source_path = os.path.abspath(source_path)
    if os.path.lexists(link_path):
        if os.path.islink(link_path) and os.path.abspath(os.readlink(link_path)) == source_path:
            print(f"Symlink already correct: {link_path} -> {source_path}")
            return
        raise FileExistsError(f"Path already exists and is not the expected link: {link_path}")

    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    os.symlink(source_path, link_path)
    print(f"Created symlink: {link_path} -> {source_path}")


if __name__ == "__main__":
    args = parse_input_parameters()
    conf = read_conf_file(args.config)
    sources_conf = read_conf_file(args.sources_config)

    create_tree(args.root, conf["tree"])

    make_symlink(sources_conf["path_input_dir"], os.path.join(args.root, "data.input", "original"))
    make_symlink(sources_conf["path_target_dir"], os.path.join(args.root, "data.target", "original"))
    make_symlink(sources_conf["path_rivers_dir"], os.path.join(args.root, "data.rivers", "rivers"))
