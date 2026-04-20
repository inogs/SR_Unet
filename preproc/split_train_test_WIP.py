import os
import random
import shutil
import sys
import json
from typing import List, Any
from alive_progress import alive_bar
import random

seed = 42


# THIS MUST BECOME GET FILE TO MOVE AND THEN YOU MOVE ALL OF THEM
# get two lists as return value: the first one contains the files to move, the second one contains the remaining files
def get_random_files(source_dir:str, percentage:float) -> (List[str], List[str]):
    
    random.seed(seed)

    file_list = os.listdir(source_dir)
    file_list = [f for f in file_list if os.path.isfile(os.path.join(source_dir, f))]

    seasons = {"winter" : [], "spring" : [], "summer" : [], "autumn" : []}
    for file_name in file_list:
        # print("file name:", file_name)
        t = int(file_name[-6:-3])
        if 0 <= t < 18:
            seasons["winter"].append(file_name)
        elif 18 <= t < 36:
            seasons["spring"].append(file_name)
        elif 36 <= t < 54:
            seasons["summer"].append(file_name)
        else:
            seasons["autumn"].append(file_name)
    # We ensure that data of each season are balanced in test and training set
    files_to_move = []
    for ssn in seasons:
        l = seasons[ssn]
        # Calculate the number of files to select -> MODIFICO PER NON RISCHIARE DI AVERE 0 FILE PER QUELLA STAGIONE, A MENO CHE LA LEN = 0
        num_files = min(len(l), max(1, int(len(l) * percentage)))
        # Randomly select the files
        random_files = random.sample(l, num_files)
        # Add files to the list
        files_to_move += random_files
    
    print(f"Selected {len(files_to_move)} files to move, which corresponds to {len(files_to_move)/len(file_list)*100:.2f}% of the total files.")
    print(f"Selected files: {files_to_move}")

    # take a second list of the remaining files
    other_files = [f for f in file_list if f not in files_to_move]
    # print number of remaining files
    print(f"Remaining files: {len(other_files)}")

    return files_to_move, other_files

def move_files(files_to_move:List[str], source_dir:str, dest_dir:str) -> None:

    os.makedirs(dest_dir, exist_ok=True)

    for file in files_to_move:
        file_src_path = os.path.join(source_dir, file)
        file_dst_path = os.path.join(dest_dir, file)

        if not os.path.isfile(src):
            raise FileNotFoundError(f"Source file does not exist: {file_src_path}")
        # end if
        
        shutil.move(file_src_path, file_dst_path)
    # end for
# end def (move_files)

def read_input_parameters():
    i = 1
    test_size = None
    data_path = None
    val = False

    
    while i < len(sys.argv):
        if sys.argv[i] == "-dp":
            if data_path != None: raise ValueError("Repeated input for data path")
            data_path = sys.argv[i+1]; i+= 2
        elif sys.argv[i] == "-ts":
            if test_size != None: raise ValueError("Repeated input for variable")
            test_size = sys.argv[i+1]; i+= 2
        elif sys.argv[i] == "-val":
            val = True
            i += 1
        else:
            i+=1
    if data_path is None: raise TypeError("Missing value for data path")
    # exit(0)
    if test_size is None: raise TypeError("You need to specify the size of the test set")
    # exit(0)

    print(f"Data path: {data_path}")
    print(f"Test size: {test_size}")
    print(f"Split train and validation set: {val}")

    return data_path, test_size, val
# end def
    

if __name__ == '__main__':
    '''
        Splits the dataset into train and test set.

        Parameters:
        -dp data path
        -ts indicates the percentage of data included in the test set. Example: for a test size of 20% write "-ts 0.2".
        -val if we have to split train and validation set 
    '''
   
    data_path, test_size, val = read_input_parameters()
    train_path = os.path.join(data_path, "var_train")
    test_path = os.path.join(data_path, "var_test")
    # check if directories exist, if not create them
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    percentage_to_move = float(test_size)

    test_files_list, train_files_list = get_random_files(data_path, percentage_to_move)


    print(f"[split_test_train with test size {test_size}] Starting execution")
    move_files(test_files_list, source_dir = data_path, dest_dir = test_path)
    move_files(train_files_list, source_dir = data_path, dest_dir = train_path)
    print(f"[split_test_train with test size {test_size}] Ending execution")
