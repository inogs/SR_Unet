import pytorch_lightning as pl
import sys
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os
from functools import reduce
import shutil


sys.path.insert(0, "./..")
from utils.data_module import ICDataModule
from models.convolutional.conv_model import ConvModel

pl.seed_everything(0, workers=True)


accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(accelerator)

class obj(object):
    def __init__(self, d):
        self._len_d = len(d)
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

    def __len__(self):
        return self._len_d


def test(data_module:ICDataModule, conf:obj, output_path:str, weight_path:str, n_var=None, loss=None, stat=None):

    if loss == None:
        loss = conf.training.loss
    if n_var == None:
        n_var = conf.n_var

    model = ConvModel.load_from_checkpoint(weight_path, stats=stat)
    model.eval()

    trainer = pl.Trainer()

    # trainer.test(model=model, datamodule=data_module, verbose=True)
    
    results = trainer.test(model=model, datamodule=data_module, verbose=True)

    output_file = os.path.join(output_path, "test_results.txt")
    with open(output_file, "w") as f:
        f.write("       Test metric             DataLoader 0\n")
        f.write("────────────────────────────────────────────\n")
        for key, value in results[0].items():
            f.write(f"{key:<20} {value}\n")


if __name__ == "__main__":
    conf_path = None
    riv = False

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-cp":
            conf_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-r":
            riv = True
            i += 1
        else:
            i += 1

    if conf_path is None:
        raise TypeError("Missing configuration path")
    

    with open(conf_path, "r") as f:
        conf = obj(json.load(f))


    train_path = conf.var_train_path
    test_path = conf.var_test_path

    main_net = conf.main_net
    loss = conf.training.loss

    weight_path = conf.resume_checkpoint_path
    output_path = conf.output_path

    stat_path = conf.stat_path

    
    test_file = os.path.basename(test_path)
    var = ".".join(test_file.split(".")[:2])
    output_path = os.path.join(conf.output_path, var)
    os.makedirs(output_path, exist_ok=True)

    config_output = os.path.join(output_path, "config_used.json")
    shutil.copy(conf_path, os.path.join(output_path, "config_used.json"))


    if stat_path is not None:
         stat = np.loadtxt(stat_path, dtype=np.float32)


    data_module = ICDataModule(
            train_path = train_path,
            test_path = test_path,
            river_train_path = conf.river_train_path if riv else None,
            river_test_path = conf.river_test_path if riv else None
            #batch_size = conf.training.train_batch_size
    )

    n_var = data_module.get_numchannels()

    print(f"[testing for dataset '{os.path.basename(test_path)}'] Starting execution")
    print(
        f"[testing for dataset '{os.path.basename(test_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
        \n[testing for variable '{os.path.basename(test_path)}'] \t-- model = {main_net} \
        \n[testing for variable '{os.path.basename(test_path)}'] \t-- loss = {loss} \
        \n[testing for variable '{os.path.basename(test_path)}'] \t-- river_info = {str(riv)} \
        \n[testing for variable '{os.path.basename(test_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    test(data_module=data_module, conf=conf, output_path=output_path,  weight_path=weight_path, n_var=n_var, loss=loss, stat=stat)
    print(f"[testing for variable '{os.path.basename(test_path)}'] Ending execution")
    print(f"metrics saved:", conf.output_path)

# if __name__== "__main__":
#     i = 1
#     conf_path = None
#     output_path = None
#     main_net = None
#     riv = False
#     loss = None
#     test_path = None
#     n_var = None
#     stat_path = None
#     train_path = None
#     stat = []

#     while i < len(sys.argv):
#         if sys.argv[i] == "-cp":
#             if conf_path != None: raise ValueError("Repeated input for configuration path")
#             conf_path = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == "-op": #check if it is possible to remove
#             if output_path != None: raise ValueError("Repeated input for output path")
#             output_path = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == "-tep":
#             test_path = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == "-trp": #check whether it is possible to remove
#             train_path = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == "-wf":
#             weight_path = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == "-loss":
#             loss = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == "-net":
#             if main_net != None: raise ValueError("Repeated input for network")
#             main_net = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == "-stat":
#             if stat_path != None: raise ValueError("Repeated input for network")
#             stat_path = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == "-r":
#             riv = True
#             i += 1
#         else:
#             i +=1

#     if conf_path is None: raise TypeError("Missing value for configuration path")
#     if output_path is None: raise TypeError("Missing value for output path")
#     if main_net is None: raise TypeError("Missing value for network")

#     with open(conf_path, 'r') as f:
#         conf = json.load(f)
#     conf = obj(conf)

#     if test_path == None:
#         test_path = conf.var_test_path
#     if not loss:
#         loss = conf.training.loss

#     # vars = []
#     test_file = os.path.basename(test_path)
#     # varnames = test_file[:test_file.find("_test_dataset")]
#     vars = [test_file.split(".")[0]]


# # modifica corretta ma lunga
#     # if stat_path is not None:
#     #     stat = []
#     #     with open(stat_path, 'r') as file:
#     #         line_elements = []
#     #         for line in file:
#     #             if line.strip():
#     #                 line_elements.append(np.float32(line))
#     #         stat.append(line_elements[0])
#     #         stat.append(line_elements[1])
#     #         stat = np.array(stat)
    
#     # mi sa che non mi serve più
#     if stat_path is not None:
#         stat = np.loadtxt(stat_path, dtype=np.float32)


#     data_module = ICDataModule(
#             train_path = train_path,
#             test_path = test_path,
#             river_train_path = conf.river_train_path if riv else None,
#             river_test_path = conf.river_test_path if riv else None,
#             batch_size = conf.training.batch_size
#     )

#     n_var = data_module.get_numchannels()

#     print(f"[testing for dataset '{os.path.basename(test_path)}'] Starting execution")
#     print(
#         f"[testing for dataset '{os.path.basename(test_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
#         \n[testing for variable '{os.path.basename(test_path)}'] \t-- model = {main_net} \
#         \n[testing for variable '{os.path.basename(test_path)}'] \t-- loss = {loss} \
#         \n[testing for variable '{os.path.basename(test_path)}'] \t-- river_info = {str(riv)} \
#         \n[testing for variable '{os.path.basename(test_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
#     )
#     test(data_module=data_module, main_net=main_net, riv_net=riv, conf=conf, output_path=output_path, test_path=test_path, weight_path=weight_path, n_var=n_var, loss=loss, stat=stat)
#     print(f"[testing for variable '{os.path.basename(test_path)}'] Ending execution")
