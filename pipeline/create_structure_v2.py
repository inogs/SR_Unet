import os
import json

# three directories, target, input and output
path_target_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/NARF.cleanup"
path_input_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/AdriaticNC"
path_river_dir = "/leonardo_scratch/large/userexternal/gzuccari/ARCHIVE/rivers"
path_main_dir = "/leonardo_scratch/large/userexternal/gzuccari/OPA_HOME_DEVELOP"


if __name__ == "__main__":
    print("Creating structure for OPA_HOME_DEVELOP")

    # create folder input_data inside path_main_dir
    path_input_data = os.path.join(path_main_dir, "data.input")
    os.makedirs(path_input_data, exist_ok=True)
    print(f"Created folder: {path_input_data}")

    # create folder target_data inside path_main_dir
    path_target_data = os.path.join(path_main_dir, "data.target")
    os.makedirs(path_target_data, exist_ok=True)
    print(f"Created folder: {path_target_data}")

    # create folder river_data inside path_main_dir
    path_river_data = os.path.join(path_main_dir, "data.rivers")
    os.makedirs(path_river_data, exist_ok=True)
    print(f"Created folder: {path_river_data}")

    # create a folder split inside path_main_dir
    path_split = os.path.join(path_main_dir, "splits")
    os.makedirs(path_split, exist_ok=True)
    print(f"Created folder: {path_split}")

    # create a conf.files inside path_main_dir
    path_conf_files = os.path.join(path_main_dir, "conf.files")
    os.makedirs(path_conf_files, exist_ok=True)
    print(f"Created folder: {path_conf_files}")

    # create a log folder inside path_main_dir
    path_log = os.path.join(path_main_dir, "log")
    os.makedirs(path_log, exist_ok=True)
    print(f"Created folder: {path_log}")

    # inside input_data create four folders: original, interpolated, logarithm, treshold, treshold.logarithm
    for folder_name in ["original", "interpolated", "converted", "treshold", "treshold.converted"]:
        path_folder = os.path.join(path_input_data, folder_name)
        os.makedirs(path_folder, exist_ok=True)
        print(f"Created folder: {path_folder}")

    # inside target_data create four folders: original, logarithm, treshold, treshold.logarithm
    for folder_name in ["original", "converted", "treshold", "treshold.converted"]:
        path_folder = os.path.join(path_target_data, folder_name)
        os.makedirs(path_folder, exist_ok=True)
        print(f"Created folder: {path_folder}")
    

    # populate data.target/original and data.input/original with symbolic links to the files in path_target_dir and path_input_dir respectively
    for file_name in os.listdir(path_target_dir):
        source_file = os.path.join(path_target_dir, file_name)
        target_file = os.path.join(path_target_data, "original", file_name)
        if not os.path.exists(target_file):
            os.symlink(source_file, target_file)
            print(f"Created symbolic link: {target_file} -> {source_file}")
    
    for file_name in os.listdir(path_input_dir):
        source_file = os.path.join(path_input_dir, file_name)
        target_file = os.path.join(path_input_data, "original", file_name)
        if not os.path.exists(target_file):
            os.symlink(source_file, target_file)
            print(f"Created symbolic link: {target_file} -> {source_file}")

    # populate data.rivers with symbolic links to the files in path_river_dir
    for file_name in os.listdir(path_river_dir):
        source_file = os.path.join(path_river_dir, file_name)
        target_file = os.path.join(path_river_data, file_name)
        if not os.path.exists(target_file):
            os.symlink(source_file, target_file)
            print(f"Created symbolic link: {target_file} -> {source_file}")
# end main
