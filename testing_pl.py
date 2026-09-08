import pytorch_lightning as pl
import sys
import numpy as np
import torch
import json
import os
import shutil
from pathlib import Path

sys.path.insert(0, "./..")
from utils.data_module import ICDataModule
from models.convolutional.conv_model import ConvModel

 # pl.seed_everything(0, workers=True)

CONFIG_FILE = (
    Path(__file__).parent
    / "conf_test_prototype.json"
)

def read_config():
    with open(CONFIG_FILE, "r") as f:
        return obj(json.load(f))

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


def test(data_module:ICDataModule, conf:obj, output_path:str, weight_path:str, stat=None, log_transform = False, threshold = None): #  n_var=None,loss=None, 


    model = ConvModel.load_from_checkpoint(weight_path, stats=stat, log_transform = log_transform, threshold = threshold, output_path = output_path)

    trainer = pl.Trainer()

    
    results = trainer.test(model=model, datamodule=data_module, verbose=False)

    ordered_keys = [
    "test_rmse",
    "test_rmse_std",
    "test_rmse_log",
    "test_rmse_log_std",
    "test_rmse_on_exp_pred_log",
    "test_exp_rmse_std",
    "test_mse",
    "test_mse_mean",
    "test_mse_std",
    "test_ssim",
    "test_ssim_std",
]


    # salvo file
    output_file = os.path.join(output_path, "test_results.txt")
    with open(output_file, "w") as f:
        f.write("       Test metric             DataLoader 0\n")
        f.write("────────────────────────────────────────────\n")

        for key in ordered_keys:
            if key in results[0]:
                f.write(f"{key:<30} {results[0][key]}\n")
        for key, value in results[0].items():
            if key not in ordered_keys:
                f.write(f"{key:<30} {value}\n")


    # stampo a video            
    print("       Test metric             DataLoader 0")
    print("────────────────────────────────────────────")

    for key in ordered_keys:
        if key in results[0]:
            print(f"{key:<30} {results[0][key]}")

    # output_file = os.path.join(output_path, "test_results.txt")
    # with open(output_file, "w") as f:
    #     f.write("       Test metric             DataLoader 0\n")
    #     f.write("────────────────────────────────────────────\n")
    #     for key, value in results[0].items():
    #         f.write(f"{key:<20} {value}\n")

    # print(results)


def main():

    conf = read_config()

    if hasattr(conf, "river_flag"):
        riv = conf.river_flag
    else:
        print("WARNING: 'river_flag' not found in the configuration. Using default value False.")
        riv = False




    pl.seed_everything(conf.seed, workers=True)

    log_transform = conf.log_transform


    test_path = conf.var_test_path

    main_net = conf.main_net
    loss = conf.training.loss

    weight_path = conf.resume_checkpoint_path
    output_path = conf.output_path

    stat_path = conf.stat_path

    test_file = os.path.basename(test_path)
    var = ".".join(test_file.split(".")[:2])


    # NB che questa threshold viene usata solo per il calcolo della rmse[log(dati+T))]
    _, var_t = var.split(".", 1)
    threshold = getattr(conf.thresholds, var_t, None)
    #     threshold = getattr(conf.thresholds, var_t)

    output_path = os.path.join(
        conf.output_path,
        var
    )

    os.makedirs(
        output_path,
        exist_ok=True
    )

    shutil.copy(
        CONFIG_FILE,
        os.path.join(output_path, "config_used.json")
    )

    stat = None
    if stat_path is not None:
        stat = np.loadtxt(
            stat_path,
            dtype=np.float32
        )

    data_module = ICDataModule(
        test_path=test_path,
        river_test_path=conf.river_test_path if riv else None
    )


    print(f"[testing for dataset '{os.path.basename(test_path)}'] Starting execution")
    print(
        f"[testing for dataset '{os.path.basename(test_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        f"\n[testing for variable '{os.path.basename(test_path)}'] \t-- model = {main_net}"
        f"\n[testing for variable '{os.path.basename(test_path)}'] \t-- loss = {loss}"
        f"\n[testing for variable '{os.path.basename(test_path)}'] \t-- river_info = {riv}"
        f"\n[testing for variable '{os.path.basename(test_path)}'] \t-- log_info = {log_transform}"
        f"\n[testing for variable '{os.path.basename(test_path)}'] \t-- threshold = {threshold}"
        f"\n[testing for dataset '{os.path.basename(test_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )

    test(
        data_module=data_module,
        conf=conf,
        output_path=output_path,
        weight_path=weight_path,
        stat=stat,
        log_transform = log_transform,
        threshold = threshold
    )

    print(f"[testing for variable '{os.path.basename(test_path)}'] Ending execution")
    print(f"metrics saved: {output_path}")


if __name__ == "__main__":
    main()