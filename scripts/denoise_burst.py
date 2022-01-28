
# -- python imports --
import numpy as np
from pathlib import Path

# -- this package --
import vnlb

# -- helper imports --
from vnlb.testing.data_loader import load_dataset
from vnlb.testing.file_io import save_images


def denoise_burst():

    # -- parse parameters --
    params = vnlb.swig.setVnlbParams(noisy.shape,sigma,params=in_params)
