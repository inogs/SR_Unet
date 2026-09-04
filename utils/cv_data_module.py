import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset
from typing import Optional
import torch.utils.data
from torch.utils.data import Subset
from pipeline.stat import (
    compute_all_file_stats,
    compute_stats_from_cached_files,
)
from pathlib import Path
from pytorch_lightning.trainer.states import TrainerFn



torch.serialization.add_safe_globals(
    [torch.utils.data.dataset.TensorDataset]
)

import os, resource, torch

import os, time

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

def _print_mem(tag):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[{tag}] pid={os.getpid()} CPU RSS: {rss_kb/(1024**2):.2f} GB")
    if torch.cuda.is_available():
        print(f"[{tag}] GPU alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")




def make_shape_even(image_tensor):
    '''
    Utility to use when either height or width of images is not even and the architecture works just for images with even shape values.
    '''
    channel, original_height, original_width = image_tensor.shape

    new_height = original_height + 1 if original_height % 2 != 0 else original_height
    new_width = original_width + 1 if original_width % 2 != 0 else original_width


    increased_image_tensor = torch.zeros(channel, new_height, new_width)

    increased_image_tensor[:, :original_height, :original_width] = image_tensor
    increased_image_tensor[:, :original_height, -1] = image_tensor[:, :, -1]
    increased_image_tensor[:, -1, :original_width] = image_tensor[:, -1, :]
    increased_image_tensor[:, -1, -1] = increased_image_tensor[:, -2, -2]

    return increased_image_tensor


class ICDataset(Dataset):
    '''
    Data coming from this custom dataset are returned as:
    (raw_data, river_data, target_value)

    '''
    def __init__(self, tensor_dataset, rivers_dataset=None, resize_to_even = False, mean_input=None, std_input=None, mean_target=None, std_target=None):
        self.ds = tensor_dataset
        self.rivers = rivers_dataset
        self.resize_to_even = resize_to_even

        self.mean_input = mean_input
        self.std_input = std_input

        self.mean_target = mean_target
        self.std_target = std_target

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x_sample, y_sample = self.ds[idx]

        # INPUT
        if self.mean_input is not None:

            valid_mask = x_sample != 0

            x_normalized = torch.zeros_like(x_sample)

            x_normalized[valid_mask] = (
                x_sample[valid_mask] - self.mean_input
            ) / self.std_input

            x_sample = x_normalized

        # TARGET
        if self.mean_target is not None:

            valid_mask = y_sample != 0

            y_normalized = torch.zeros_like(y_sample)

            y_normalized[valid_mask] = (
                y_sample[valid_mask] - self.mean_target
            ) / self.std_target

            y_sample = y_normalized

        
        if self.resize_to_even and (x_sample.shape[-2] % 2 != 0 or x_sample.shape[-1] % 2 != 0):
            x_sample = make_shape_even(x_sample)
            y_sample = make_shape_even(y_sample)
        
        if self.rivers is not None:
            river_sample = self.rivers[idx]
            return x_sample, river_sample, y_sample

        return x_sample, y_sample



# def compute_stats_from_indices(dataset, indices):
#     """
#     Calcola mean e std di INPUT e TARGET usando solo i campioni
#     del training fold, ignorando i punti masked (= 0).

#     Non concatena tutti i valori in RAM.
#     """

#     def update_stats(count, mean, M2, values):
#         if values.numel() == 0:
#             return count, mean, M2

#         values = values.float()

#         n = values.numel()
#         batch_mean = values.mean().item()
#         batch_var = values.var(unbiased=False).item()
#         batch_M2 = batch_var * n

#         if count == 0:
#             return n, batch_mean, batch_M2

#         delta = batch_mean - mean
#         new_count = count + n

#         new_mean = mean + delta * n / new_count

#         new_M2 = (
#             M2
#             + batch_M2
#             + delta**2 * count * n / new_count
#         )

#         return new_count, new_mean, new_M2

#     # INPUT
#     count_input = 0
#     mean_input = 0.0
#     M2_input = 0.0

#     # TARGET
#     count_target = 0
#     mean_target = 0.0
#     M2_target = 0.0

#     for i,idx in enumerate(indices):

        
#         if i % 20 == 0:
#             print(
#                 f">>> Stats progress: {i}/{len(indices)} samples",
#                 flush=True
#             )

#         x_sample, y_sample = dataset.ds[idx]

#         x_sample = x_sample.float()
#         y_sample = y_sample.float()

#         # escludo i punti masked
#         x_valid = x_sample[x_sample != 0]
#         y_valid = y_sample[y_sample != 0]

#         count_input, mean_input, M2_input = update_stats(
#             count_input,
#             mean_input,
#             M2_input,
#             x_valid
#         )

#         count_target, mean_target, M2_target = update_stats(
#             count_target,
#             mean_target,
#             M2_target,
#             y_valid
#         )


#     if count_input == 0:
#         raise ValueError("No valid input values found")

#     if count_target == 0:
#         raise ValueError("No valid target values found")

#     std_input = np.sqrt(M2_input / count_input)
#     std_target = np.sqrt(M2_target / count_target)

#     return mean_input, std_input, mean_target, std_target


class ICDataModule(pl.LightningDataModule):

    def __init__(self, dataset, test_path:Optional[str]=None, river_test_path:Optional[str]=None, train_batch_size:int=32, val_batch_size:int=32, train_shuffle:bool=False, val_shuffle:bool=False, test_shuffle:bool=False,  train_idx=None, val_idx=None, resize_to_even:bool=False, stats_path=None, train_files_txt=None, target_train_files_txt=None):
        super(ICDataModule, self).__init__()
        self.dataset = dataset
        self.stats_path = stats_path

        self.test_path=test_path
        self.river_test_path=river_test_path

        self.train_files_txt = train_files_txt
        self.target_train_files_txt = target_train_files_txt

        self.train_batch_size=train_batch_size
        self.val_batch_size=val_batch_size

        self.train_shuffle=train_shuffle
        self.val_shuffle=val_shuffle
        self.test_shuffle=test_shuffle

        self.resize_to_even = resize_to_even

        self.train_idx = train_idx
        self.val_idx = val_idx

    # def prepare_data(self):

    #     print(">>> ENTER prepare_data", flush=True)

    #     if os.path.exists(self.stats_path):
    #         print(
    #             f">>> Statistics already exist: {self.stats_path}",
    #             flush=True
    #         )
    #         return

    #     # leggo le liste complete input/target
    #     with open(self.train_files_txt) as f:
    #         files = [line.strip() for line in f if line.strip()]

    #     with open(self.target_train_files_txt) as f:
    #         target_files = [line.strip() for line in f if line.strip()]

    #     assert len(files) == len(target_files) == len(self.dataset)

    #     # prendo SOLO i file appartenenti al training del fold
    #     fold_files = [
    #         files[idx]
    #         for idx in self.train_idx
    #     ]

    #     fold_target_files = [
    #         target_files[idx]
    #         for idx in self.train_idx
    #     ]

    #     # creo i txt temporanei del fold
    #     output_dir = os.path.dirname(self.stats_path)
    #     stats_id = Path(self.stats_path).stem


    #     # ricavo fold + job_id da stats_path
    #     run_id = Path(self.stats_path).stem.replace("stats_", "")

    #     input_fold_txt = os.path.join(
    #         output_dir,
    #         f"train_input_{run_id}.txt"
    #     )

    #     target_fold_txt = os.path.join(
    #         output_dir,
    #         f"train_target_{run_id}.txt"
    #     )

    #     with open(input_fold_txt, "w") as f:
    #         f.write("\n".join(fold_files) + "\n")

    #     with open(target_fold_txt, "w") as f:
    #         f.write("\n".join(fold_target_files) + "\n")

    #     # nomi variabili NetCDF
    #     var_input = Path(fold_files[0]).name.split("_")[0]
    #     var_target = Path(fold_target_files[0]).name.split("_")[0]

    #     # numero CPU assegnate dal job
    #     jobs = int(
    #         os.environ.get("SLURM_CPUS_PER_TASK", "1")
    #     )

    #     print(
    #         f">>> Computing statistics with {jobs} workers",
    #         flush=True
    #     )

    #     mean_input, std_input, *_ = compute_list_stats(
    #         input_fold_txt,
    #         var_input,
    #         jobs
    #     )

    #     mean_target, std_target, *_ = compute_list_stats(
    #         target_fold_txt,
    #         var_target,
    #         jobs
    #     )

    #     if std_input == 0:
    #         raise ValueError("Input std is zero")

    #     if std_target == 0:
    #         raise ValueError("Target std is zero")

    #     np.savetxt(
    #         self.stats_path,
    #         [
    #             mean_input,
    #             std_input,
    #             mean_target,
    #             std_target
    #         ]
    #     )

    #     print(
    #         f">>> Statistics computed:\n"
    #         f"input  mean={mean_input:.6e}, std={std_input:.6e}\n"
    #         f"target mean={mean_target:.6e}, std={std_target:.6e}",
    #         flush=True
    #     )

    def prepare_data(self):

        print(">>> ENTER prepare_data", flush=True)

        # ==========================================================
        # Se le statistiche FINALI di questo fold esistono già,
        # non devo fare assolutamente niente
        # ==========================================================

        if os.path.exists(self.stats_path):
            print(
                f">>> Fold statistics already exist: {self.stats_path}",
                flush=True
            )
            return


        # ==========================================================
        # 1. LEGGO LE LISTE COMPLETE DEL DATASET CV
        # ==========================================================

        with open(self.train_files_txt) as f:
            files = [
                line.strip()
                for line in f
                if line.strip()
            ]

        with open(self.target_train_files_txt) as f:
            target_files = [
                line.strip()
                for line in f
                if line.strip()
            ]

        assert len(files) == len(target_files) == len(self.dataset)


        # ==========================================================
        # 2. NOME VARIABILI
        # ==========================================================

        var_input = Path(files[0]).name.split("_")[0]
        var_target = Path(target_files[0]).name.split("_")[0]


        # ==========================================================
        # 3. NUMERO WORKER
        # ==========================================================

        jobs = int(
            os.environ.get("SLURM_CPUS_PER_TASK", "1")
        )

        print(
            f">>> Statistics workers: {jobs}",
            flush=True
        )


        # ==========================================================
        # 4. PATH DELLA CACHE
        #
        # stats_path esempio:
        # stats_fold0_54156858.txt
        #
        # cache comune a TUTTI i fold dello stesso job
        # ==========================================================

        output_dir = os.path.dirname(self.stats_path)

        stats_name = Path(self.stats_path).stem

        # stats_fold0_54156858 -> 54156858
        job_id = stats_name.rsplit("_", 1)[-1]

        input_cache_path = os.path.join(
            output_dir,
            f"file_stats_input_{job_id}.npz"
        )

        target_cache_path = os.path.join(
            output_dir,
            f"file_stats_target_{job_id}.npz"
        )


        # ==========================================================
        # 5. INPUT
        #
        # SOLO IL PRIMO FOLD entra nel compute.
        # Gli altri fold fanno np.load().
        # ==========================================================

        if not os.path.exists(input_cache_path):

            print(
                ">>> INPUT cache not found."
                " Computing per-file statistics ONCE...",
                flush=True
            )
            
            t0 = time.time()
            input_mus, input_vars = compute_all_file_stats(
                files,
                var_input,
                jobs
            )

            np.savez(
                input_cache_path,
                mus=input_mus,
                vars=input_vars
            )

            print(
                f">>> INPUT cache saved: {input_cache_path}",
                flush=True
            )

        else:

            print(
                f">>> Loading INPUT cache: {input_cache_path}",
                flush=True
            )

            with np.load(input_cache_path) as cache:
                input_mus = cache["mus"]
                input_vars = cache["vars"]


        # ==========================================================
        # 6. TARGET
        # ==========================================================

        if not os.path.exists(target_cache_path):

            print(
                ">>> TARGET cache not found."
                " Computing per-file statistics ONCE...",
                flush=True
            )   
            
            target_mus, target_vars = compute_all_file_stats(
                target_files,
                var_target,
                jobs
            )

            np.savez(
                target_cache_path,
                mus=target_mus,
                vars=target_vars
            )

            print(
                f">>> TOTAL CACHE TIME: {(time.time() - t0) / 60:.2f} minutes",
                flush=True
            )

        else:

            print(
                f">>> Loading TARGET cache: {target_cache_path}",
                flush=True
            )

            with np.load(target_cache_path) as cache:
                target_mus = cache["mus"]
                target_vars = cache["vars"]


        # 7. ORA USO SOLO train_idx DEL FOLD CORRENTE
        # Questa operazione non apre nessun NetCDF

        mean_input, std_input = compute_stats_from_cached_files(
            input_mus,
            input_vars,
            self.train_idx
        )

        mean_target, std_target = compute_stats_from_cached_files(
            target_mus,
            target_vars,
            self.train_idx
        )


        # 8. CONTROLLI

        if std_input == 0:
            raise ValueError("Input std is zero")

        if std_target == 0:
            raise ValueError("Target std is zero")


        # 9. SALVO LE STATISTICHE FINALI DI QUESTO FOLD

        np.savetxt(
            self.stats_path,
            [
                mean_input,
                std_input,
                mean_target,
                std_target
            ]
        )

        print(
            f">>> Fold statistics computed:\n"
            f"input  mean={mean_input:.6e}, std={std_input:.6e}\n"
            f"target mean={mean_target:.6e}, std={std_target:.6e}",
            flush=True
        )
                

            
    def setup(self, stage=None):
        print(f">>> ENTER setup, stage={stage}", flush=True)
        mean_input, std_input, mean_target, std_target = np.loadtxt(
            self.stats_path
        )

        normalized_dataset = ICDataset(
            tensor_dataset=self.dataset.ds,
            rivers_dataset=self.dataset.rivers,
            resize_to_even=self.resize_to_even,
            mean_input=mean_input,
            std_input=std_input,
            mean_target=mean_target,
            std_target=std_target,
        )

        if stage == "fit":
            self.train_ds = Subset(normalized_dataset, self.train_idx)
            self.val_ds = Subset(normalized_dataset, self.val_idx)

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            if self.test_path is None:
                raise ValueError("test_path must be specified for test")
            test_ds = torch.load(self.test_path)
            if self.river_test_path != None:
                rivers_ds = torch.load(self.river_test_path)
            else:
                rivers_ds = None
            self.test_ds = ICDataset(tensor_dataset=test_ds, rivers_dataset=rivers_ds, resize_to_even=self.resize_to_even)

        # commento Anna: in teoria con get_numchannels carico di nuovo tutto il dataset (torch.load) in memoria -> basterebbe aggiungere mmap_mode = 'r'? o contare il numero di var quando si fanno i load dei dataset per training, o quando si costruisce i dataset
        # def get_numchannels(self):
        #     test_dataset = torch.load(self.test_path)
        #     test_ds = ICDataset(tensor_dataset=test_dataset)
        #     return np.array(test_ds).shape[2]

    def get_numchannels(self):
        sample = self.dataset[0]

        if len(sample) == 2:
            x, _ = sample
        else:
            x, _, _ = sample

        return x.shape[0]


    def train_dataloader(self):
        # return torch.utils.data.DataLoader(self.train_ds,
        #                                    batch_size=self.train_batch_size,
        #                                    num_workers=8,
        #                                    pin_memory=True,
        #                                    shuffle = True
        #                                    )
        return torch.utils.data.DataLoader(
                                            self.train_ds,
                                            batch_size=self.train_batch_size,
                                            num_workers=8,
                                            pin_memory=True,
                                            shuffle=self.train_shuffle,
                                            persistent_workers=True,
                                            )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds,
                                           batch_size=self.val_batch_size,
                                           num_workers=8,
                                           pin_memory=True,
                                           shuffle = self.val_shuffle)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds,
                                           batch_size=1,
                                           num_workers=8,
                                           pin_memory=True,
                                           shuffle = self.test_shuffle)