import os
import sys

print("=== ENV SMOKE TEST ===", flush=True)

print("python:", sys.version.replace("\n", " "), flush=True)
print("executable:", sys.executable, flush=True)

# Ensure imports work from repo root
sys.path.insert(0, os.path.abspath("."))
print("sys.path[0] =", sys.path[0], flush=True)

def mark(s):
    print(s, flush=True)

def where(mod):
    return getattr(mod, "__file__", "<built-in>")

mark("import numpy ...")
import numpy as np
mark(f"OK numpy  ({where(np)})")

mark("import scipy ...")
import scipy
mark(f"OK scipy  ({where(scipy)})")

mark("import netCDF4 ...")
import netCDF4
mark(f"OK netCDF4 ({where(netCDF4)})")

mark("import matplotlib ...")
import matplotlib
mark(f"OK matplotlib ({where(matplotlib)})")
try:
    import matplotlib.pyplot as plt
    mark(f"matplotlib backend: {matplotlib.get_backend()}")
except Exception as e:
    mark("matplotlib pyplot/backend check failed: " + repr(e))

mark("import cmocean ...")
import cmocean
mark(f"OK cmocean ({where(cmocean)})")

mark("import torch ...")
import torch
mark(f"OK torch ({where(torch)})")

mark("printing versions ...")
print("numpy:", np.__version__, flush=True)
print("scipy:", scipy.__version__, flush=True)
print("netCDF4:", netCDF4.__version__, flush=True)
print("matplotlib:", matplotlib.__version__, flush=True)

print(
    "torch:", torch.__version__,
    "cuda_available:", torch.cuda.is_available(),
    "torch_cuda:", torch.version.cuda,
    "cudnn:", torch.backends.cudnn.version(),
    flush=True
)

if torch.cuda.is_available():
    try:
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
    except Exception as e:
        mark("GPU name check failed: " + repr(e))

# Lightning import compatibility
mark("import pytorch_lightning / lightning ...")
pl = None
try:
    import pytorch_lightning as pl
    mark("OK pytorch_lightning")
    print("pytorch_lightning:", pl.__version__, flush=True)
except Exception as e:
    mark("pytorch_lightning import failed: " + repr(e))
    try:
        import lightning as L
        mark("OK lightning")
        print("lightning:", L.__version__, flush=True)
        pl = L
    except Exception as e2:
        mark("lightning import failed: " + repr(e2))

mark("\n=== REPO IMPORT SMOKE TEST ===")
try:
    from utils.data_module import ICDataModule
    mark("OK: from utils.data_module import ICDataModule")
except Exception as e:
    mark("FAIL importing ICDataModule: " + repr(e))

try:
    from models.convolutional.conv_model import ConvModel
    mark("OK: from models.convolutional.conv_model import ConvModel")
except Exception as e:
    mark("FAIL importing ConvModel: " + repr(e))

mark("\n=== LIGHTNING TRAINER SMOKE TEST ===")
if pl is None:
    mark("SKIP Trainer test (no lightning import succeeded)")
else:
    try:
        trainer = pl.Trainer(
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False
        )
        mark("OK: Trainer constructed")
    except Exception as e:
        mark("FAIL constructing Trainer: " + repr(e))

mark("\nDONE.")
