# local imports
from utils.data_module import ICDataModule
from models.convolutional.conv_model import ConvModel

# basic imports
import json
import os
import time
import sys
import argparse
from pathlib import Path
from functools import reduce
from types import SimpleNamespace

# torch and lightning imports
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from lightning_fabric.utilities.rank_zero import rank_zero_only

# from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

# torch and pl lightning settings
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
pl.seed_everything(0, workers=True)

# accelerator and device settings
accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(accelerator)

@rank_zero_only
def rprint(*args, **kwargs):
    print(*args, **kwargs)


def parse_input_parameters():
    parser = argparse.ArgumentParser(
        description="Train the production SR-UNet model from a config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "conf_train.json"),
        help="Path to configuration file.",
    )
    return parser.parse_args()


def validate_conf_file_path(conf_path):
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")
    if not os.path.isfile(conf_path):
        raise ValueError(f"Configuration path is not a file: {conf_path}")
    print("Configuration path check: passed")
    print(f"    Configuration path: {conf_path}")


def read_conf_file(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f, object_hook=lambda data: SimpleNamespace(**data))


def _require_fields(conf, field_names, prefix="conf"):
    for field_name in field_names:
        if not hasattr(conf, field_name):
            raise AttributeError(f"Missing configuration field: {prefix}.{field_name}")


def _validate_existing_path(conf, field_name):
    field_value = getattr(conf, field_name)
    if not isinstance(field_value, str) or not field_value.strip():
        raise ValueError(f"Configuration field must be a non-empty path string: {field_name}")
    if not os.path.exists(field_value):
        raise FileNotFoundError(f"Configured path does not exist: {field_name}={field_value}")
    print(f"    Path {field_value} exists, check passed")


def _validate_positive_int(conf, field_name):
    field_value = getattr(conf, field_name)
    if not isinstance(field_value, int) or field_value <= 0:
        raise ValueError(f"Configuration field must be a positive integer: {field_name}")


def validate_conf(conf):
    required_fields = [
        "n_var",
        "n_riv",
        "river_flag",
        "seed",
        "main_net",
        "data_dim",
        "n_gpus",
        "output_path",
        "var_train_path",
        "var_test_path",
        "river_train_path",
        "river_test_path",
        "training",
    ]
    required_training_fields = [
        "loss",
        "lr",
        "max_epochs",
        "precision",
        "patience",
        "batch_size",
        "accumulate_grad_batches",
    ]

    _require_fields(conf, required_fields)
    _require_fields(conf.training, required_training_fields, prefix="conf.training")

    _validate_positive_int(conf, "n_var")
    _validate_positive_int(conf, "n_gpus")

    if not isinstance(conf.n_riv, int) or conf.n_riv < 0:
        raise ValueError("Configuration field must be a non-negative integer: n_riv")
    if not isinstance(conf.river_flag, bool):
        raise ValueError("Configuration field must be boolean: river_flag")
    if not isinstance(conf.seed, int):
        raise ValueError("Configuration field must be an integer: seed")
    if not isinstance(conf.main_net, str) or not conf.main_net.strip():
        raise ValueError("Configuration field must be a non-empty string: main_net")

    if not vars(conf.data_dim):
        raise ValueError("Configuration field data_dim must contain at least one dimension")
    for field_name, field_value in vars(conf.data_dim).items():
        if not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"Configuration field must be a positive integer: data_dim.{field_name}")

    if conf.training.loss not in ["mse", "rmse", "perceptual"]:
        raise ValueError("Configuration field training.loss must be one of: mse, rmse, perceptual")
    if conf.training.precision not in ["32-true", "16-mixed", "bf16-mixed"]:
        raise ValueError("Configuration field training.precision must be one of: 32-true, 16-mixed, bf16-mixed")
    if not isinstance(conf.training.lr, (int, float)) or conf.training.lr <= 0:
        raise ValueError("Configuration field must be a positive number: training.lr")
    for field_name in ["max_epochs", "patience", "batch_size", "accumulate_grad_batches"]:
        field_value = getattr(conf.training, field_name)
        if not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"Configuration field must be a positive integer: training.{field_name}")

    if not isinstance(conf.output_path, str) or not conf.output_path.strip():
        raise ValueError("Configuration field must be a non-empty path string: output_path")
    _validate_existing_path(conf, "var_train_path")
    _validate_existing_path(conf, "var_test_path")

    if conf.river_flag:
        _validate_existing_path(conf, "river_train_path")
        _validate_existing_path(conf, "river_test_path")

    print("Configuration file check: passed")
    for field_name, field_value in sorted(vars(conf).items()):
        print(f"    {field_name}: {field_value}")

class MetricsLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.val_psnr = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        metrics = trainer.callback_metrics
        # print(metrics.keys())
        train_loss = metrics.get('train_loss')
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
            print(f"Epoch {epoch}: train loss = {train_loss:.4f}")
        else:
            self.train_losses.append(float('nan'))


    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        metrics = trainer.callback_metrics
        # print(metrics.keys())
        val_loss = metrics.get('val_loss')
        val_psnr = metrics.get('val_psnr')
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        else:
            self.val_losses.append(float('nan'))
        
        if val_psnr is not None:
            val_psnr = val_psnr.item()
            self.val_psnr.append(val_psnr)
        else:
            val_psnr = float('nan')
            self.val_psnr.append(val_psnr)
        
        print(f"Epoch {epoch}: val loss = {val_loss:.4f}, psnr = {val_psnr:.4f}")

    
    def save_txt(self, filepath="loss_log.txt"):
        val = list(self.val_losses)
        tra = list(self.train_losses)
        if len(val) > len(tra):
            val = val[1:]

        with open(filepath, "w") as f:
            f.write("epoch\tval_loss\ttrain_loss\n")
            for i in range(len(tra)):
                f.write(f"{i+1}\t{val[i]:.6f}\t{tra[i]:.6f}\n")

    def plot(self):
        val = list(self.val_losses)
        tra = list(self.train_losses)
        if len(val) > len(tra):
            val = val[1:]

        epochs = range(1, len(tra)+1)
        plt.figure(figsize=(10,5))
        plt.plot(epochs, tra, label='Train Loss')
        plt.plot(epochs, val, label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("Validation and Training Loss trends during Epochs")
        plt.legend()
        plt.grid(True)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig("loss_plot.png")
        plt.close()

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)


def _rss_gb_linux():
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass
    return None

def mem(tag=""):
    rss_gb = _rss_gb_linux()
    rss = f"{rss_gb:.2f}GB" if rss_gb is not None else "?"
    try:
        import torch
        if torch.cuda.is_available():
            d = torch.cuda.current_device()
            ga = torch.cuda.memory_allocated(d) / (1024**3)
            gr = torch.cuda.memory_reserved(d) / (1024**3)
            gpu = f"GPU{d} alloc={ga:.2f}GB reserv={gr:.2f}GB"
        else:
            gpu = "CUDA=off"
    except Exception as e:
        gpu = f"CUDA=? ({type(e).__name__})"
    print(f"[MEM] {time.strftime('%F %T')} {tag} :: RSS={rss} | {gpu}", flush=True)

def train(data_module:ICDataModule,conf:obj):

    model = ConvModel(
                    main_net=conf.main_net,
                    n_dimensions=len(vars(conf.data_dim)),
                    riv_net = conf.river_flag,
                    loss = conf.training.loss,
                    num_channels=conf.n_var,
                    #riv_in_dim = conf.n_riv,
                    #riv_out_dim = reduce(lambda x, y: x * y, vars(conf.data_dim).values()),
                    lr = conf.training.lr
            )

    print("output in train", reduce(lambda x, y: x * y, vars(conf.data_dim).values()))
    print(vars(conf.data_dim))
    train_file = os.path.splitext(os.path.basename(conf.var_train_path))[0]
    best_filename = f'best_{model.name}_{train_file}'

    tb_logger = loggers.TensorBoardLogger(save_dir="./")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,  # Save the model with the lowest validation loss
        dirpath=conf.output_path,
        filename=best_filename
    )

    # flag = False
    # # terminate program here if flag is false
    # if flag == False:
    #     print("Terminating program")
    #     sys.exit(0)
    
    # MODIFICA
    metrics_logger = MetricsLogger()

    # trainer = pl.Trainer(detect_anomaly=False, accelerator=accelerator, strategy="ddp_find_unused_parameters_true", log_every_n_steps=20, max_epochs=conf.training.max_epochs, callbacks=[early_stop_callback, checkpoint_callback], logger=tb_logger, check_val_every_n_epoch=1)
    trainer = pl.Trainer(
        accelerator="gpu",
        # precision="bf16-mixed",
        # precision="32-true",
        precision=conf.training.precision, # possible values: "32-true", "16-mixed", "bf16-mixed"
        devices=conf.n_gpus,
        strategy="ddp",
        log_every_n_steps=20,
        max_epochs=conf.training.max_epochs,
        callbacks=[checkpoint_callback, metrics_logger],
        logger=tb_logger,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=conf.training.accumulate_grad_batches # possible values: 1, 2, 4, 8, ... (effective batch size = batch_size * accumulate_grad_batches * n_gpus)
    )

    trainer.fit(model, datamodule=data_module)

    metrics_logger.save_txt()
    metrics_logger.plot()



if __name__== "__main__":
    """File for the training of the neural network.
    It must be used after the dataset construction.
    Inputs:
        -cp (str): complete path to the configuration file, giving the parameters
    """

    args = parse_input_parameters()
    validate_conf_file_path(args.config)
    conf = read_conf_file(args.config)
    validate_conf(conf)

    # print all the fields in the conf object
    rprint("n_var: ",conf.n_var)
    rprint("n_riv: ",conf.n_riv)
    rprint("river flag:", conf.river_flag)
    rprint("seed: ",conf.seed)
    rprint("main net:", conf.main_net)

    # paths
    rprint("train path:",conf.var_train_path)
    rprint("test path:",conf.var_test_path)
    rprint("river train path:",conf.river_train_path)
    rprint("river test path:",conf.river_test_path)

    # devices
    rprint("n_gpus (user set):", conf.n_gpus)

    # training parameters
    rprint("training loss:", conf.training.loss)
    rprint("training learning rate:", conf.training.lr)
    rprint("training max_epochs:", conf.training.max_epochs)
    rprint("training precision:", conf.training.precision)
    rprint("training patience:", conf.training.patience)
    rprint("training batch size (from input):", conf.training.batch_size)
    rprint("training accumulate_grad_batches (from input):", conf.training.accumulate_grad_batches)
    rprint("effective batch size (batch_size * accumulate_grad_batches * n_gpus):",
        conf.training.batch_size *
        conf.training.accumulate_grad_batches *
        conf.n_gpus)

    data_module = ICDataModule(
            train_path = conf.var_train_path,
            test_path = conf.var_test_path,
            river_train_path = conf.river_train_path if conf.river_flag else None,
            river_test_path = conf.river_test_path if conf.river_flag else None,
            batch_size = conf.training.batch_size
    )

    n_var = data_module.get_numchannels()
    rprint("n_var from train dataset =", n_var)
    # if n_var != conf.n_var: exit code with error message
    if n_var != conf.n_var:
        print(f"Warning: n_var in conf ({conf.n_var}) does not match number of channels in dataset ({n_var}). Using n_var = {n_var} from dataset.")
        exit(1)
    # end if

    # put a timer here to check the time taken by the training
    t0 = time.time()
    rprint(f"[training for dataset '{os.path.basename(conf.var_train_path)}'] Starting execution")
    rprint(
        f"[training for dataset '{os.path.basename(conf.var_train_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
        \n[training for variable '{os.path.basename(conf.var_train_path)}'] \t-- max_epochs = {conf.training.max_epochs} \
        \n[training for variable '{os.path.basename(conf.var_train_path)}'] \t-- model = {conf.main_net} \
        \n[training for variable '{os.path.basename(conf.var_train_path)}'] \t-- patience = {conf.training.patience} \
        \n[training for variable '{os.path.basename(conf.var_train_path)}'] \t-- lr = {conf.training.lr} \
        \n[training for variable '{os.path.basename(conf.var_train_path)}'] \t-- loss = {conf.training.loss} \
        \n[training for variable '{os.path.basename(conf.var_train_path)}'] \t-- river_info = {conf.river_flag} \
        \n[training for variable '{os.path.basename(conf.var_train_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    train(data_module=data_module, conf=conf)
    rprint(f"[training for variable '{os.path.basename(conf.var_train_path)}'] Ending execution")
    t1 = time.time()
    rprint(f"[training for variable '{os.path.basename(conf.var_train_path)}'] Total time taken: {t1-t0:.2f} seconds")
