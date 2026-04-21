import os
import shutil
import sys
import json
from typing import List, Any

def move_files(files_to_move:List[str], source_dir:str, dest_dir:str) -> None:

    os.makedirs(dest_dir, exist_ok=True)

    for file in files_to_move:
        file_src_path = os.path.join(source_dir, file)
        file_dst_path = os.path.join(dest_dir, file)

        if not os.path.isfile(file_src_path):
            raise FileNotFoundError(f"Source file does not exist: {file_src_path}")
        # end if
        
        shutil.move(file_src_path, file_dst_path)
    # end for
# end def (move_files)

def read_input_parameters():
    # get only the data path
    i = 1
    data_path = None
    
    while i < len(sys.argv):
        if sys.argv[i] == "-dp":
            if data_path != None: raise ValueError("Repeated input for data path")
            data_path = sys.argv[i+1]; i+= 2
    if data_path is None: raise TypeError("Missing value for data path")

    print(f"Data path: {data_path}")

    return data_path
# end def


    

if __name__ == '__main__':
    '''
        Splits the dataset into train and test set.

        Parameters:
        -dp data path
    '''
   
    data_path = read_input_parameters()
    train_path = os.path.join(data_path, "var_train")
    test_path = os.path.join(data_path, "var_test")
    # check if directories exist, if not return error and exit
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Directory {train_path} does not exist")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Directory {test_path} does not exist")
    
    # move all files from train and test directory to the data path
    print("Moving files from train and test directories to data path...")
    # print to file train an test file path lists

    # print_file_paths(os.listdir(train_path), output_file_path=data_path, output_file="train_files.txt")
    # print_file_paths(os.listdir(test_path), output_file_path=data_path, output_file="test_files.txt")

    move_files(os.listdir(train_path), source_dir = train_path, dest_dir = data_path)
    move_files(os.listdir(test_path), source_dir = test_path, dest_dir = data_path)
    print("Files moved successfully.")
    # remove the train and test directories
    os.rmdir(train_path)
    os.rmdir(test_path)
# end if (main)