import os
import random
import shutil
import sys
import json
from typing import List, Any
from alive_progress import alive_bar
import random

seed = 42

# Lo script vuole mantenere corrispondenza temporale tra tutte le variabili, quindi: Seleziona file solo da una variabile di riferimento (var_list[0]), Usa quei file selezionati anche per tutte le altre variabili e per i river

def confirm_split(files_to_move):
    print("\n==============================")
    print("ATTENZIONE: stai per splittare il dataset.")
    print(f"Numero di timestamp selezionati: {len(files_to_move)}")
    print("I file verranno SPOSTATI")
    print("==============================")

    answer = input("Sei sicuro di voler continuare? (yes/no): ")

    if answer.lower() not in ["yes", "y"]:
        print("Operazione annullata.")
        sys.exit(0)

    print("Conferma ricevuta. Procedo con lo split.\n")

# THIS MUST BECOME GET FILE TO MOVE AND THEN YOU MOVE ALL OF THEM
def get_random_files(source_dir:str, percentage:float, reference_var:str) -> List[str]:

# AGGIUNGO SEED PER LA RIPRODUCIBILITA'
    random.seed(seed)

    '''
        Sample a list of files to move from the source directory to the folder of the test set, ensuring that each season is equally represented.

        Args:
            source_dir (str): source directory
            percentage (float): percentage of data to be sampled
            refernce_var (str): reference var (folder) from which we sample files randomly
        Returns:
            List sampled file names (List[str])
    '''
    # We look at the reference_var folder to select random time instants for the test set
    source_dir = os.path.join(source_dir, reference_var)
    file_list = os.listdir(source_dir)
    # Divide files in the 4 seasons
    seasons = {"winter" : [], "spring" : [], "summer" : [], "autumn" : []}
    for file_name in file_list:
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
    return files_to_move

def move_files(files_to_move:List[str], source_dir:str, dest_dir:str, var_name:str, is_cms:bool, bar:Any) -> None:
    '''
        Moves all files related to the names in the files_to_move list to the destination directory.

        Args:
            files_to_move (List[str]): list of file names related to the files to be moved.
            source_dir (str): source directory.
            dest_dir (str): destination directory.
            var_name (str): name of the biogeochemical variable in the source directory.
            is_cms (bool): whether the variable name follows the CMS nomeclature ore the OGS one.
        Returns:
            List sampled file names (List[str])
    '''
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    file_list = os.listdir(source_dir)
    # Iterate over the files in the first source directory
    for file in files_to_move:
        # Extract the corresponding file name
        name_end = file[4:] if is_cms else file[4:].replace("-", "_")
        if var_name == "river":
            name_end = name_end.replace("nc", "txt")
        reference_name =  var_name + "_" + name_end
        if (reference_name in file_list):
            source_path = os.path.join(source_dir, reference_name)
            # Move file with name reference_name to the destination directory
            shutil.move(source_path, dest_dir)
            bar()

if __name__ == '__main__':
    '''
        Splits the dataset into train and test set.

        Parameters:
        -dp data path
        -ts indicates the percentage of data included in the test set. Example: for a test size of 20% write "-ts 0.2".
        -val if we have to split train and validation set 
    '''
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
    if test_size is None: raise TypeError("You need to specify the size of the test set")

    map_path = os.path.join(data_path, "cms2ogs.json")

    with open(map_path, 'r') as f:
        cms2ogs_map = json.load(f)

    var_list = list(cms2ogs_map.keys())
    
    

    
    if val:
        cms_path_3D = os.path.join(data_path, "iCMS_nc") # train copernicus 
        ogs_path_3D = os.path.join(data_path, "NARF_nc") # train ogs cadeau
        cms_test_path_3D = os.path.join(data_path,"nc_iCMS_val")
        ogs_test_path_3D = os.path.join(data_path, "nc_OGS_val")

    # mancano i dataset interi da cui fare split train e test (abbiamo solo AdriatticNC che osno i copernicus ma che non sono interpolati, e mancano i Cadeau)
    # else: 
    #     cms_path_3D = os.path.join(data_path, "iCMS_nc")
    #     ogs_path_3D = os.path.join(data_path, "NARF_nc")
    #     cms_test_path_3D = os.path.join(data_path,"nc_iCMS_test")
    #     ogs_test_path_3D = os.path.join(data_path, "nc_OGS_test")


    percentage_to_move = float(test_size)

    # Get files to move for for one of the biogeochemical variables
    #files_to_move = get_random_files(cms_path_3D, percentage_to_move, var_list[0])
    #MODIFICA TOLGO CASO SURFACE
    reference_path = cms_path_3D
    files_to_move = get_random_files(reference_path, percentage_to_move, var_list[0])

    # ti chiede conferma se vuoi procedere con lo splittamento dei datasets
    # NB. DA TOGLIERE SE SI LANCIA CON SLURM! 
    confirm_split(files_to_move)

    print(f"[split_test_train with test size {test_size}] Starting execution")
    with alive_bar(0, title=f"Moving files...") as bar:
        for var in var_list:

            # Splitting 3D data
            source_dir_cms = os.path.join(cms_path_3D, var)
            source_dir_ogs = os.path.join(ogs_path_3D, cms2ogs_map[var])
            dest_dir_cms = os.path.join(cms_test_path_3D, var)
            dest_dir_ogs = os.path.join(ogs_test_path_3D, cms2ogs_map[var])

            move_files(files_to_move, source_dir_cms, dest_dir_cms, var, True, bar)
            move_files(files_to_move, source_dir_ogs, dest_dir_ogs, cms2ogs_map[var], False, bar)
            

        # Splitting rivers data
        if val:
            source_dir_riv = os.path.join(data_path, "rivers", "rivers_train")
            dest_dir_riv = os.path.join(data_path, "rivers", "vector_val")
        else:
            source_dir_riv = os.path.join(data_path, "rivers", "vector")
            dest_dir_riv = os.path.join(data_path, "rivers", "vector_test")
        move_files(files_to_move, source_dir_riv, dest_dir_riv, "river", False, bar)
    print(f"[split_test_train with test size {test_size}] Ending execution")
