# -- python imports --
from PIL import Image
import numpy as np
import numpy.random as npr
from einops import rearrange,repeat

# -- pytorch imports --
import torch
import torchvision.utils as tvUtils


# -- import vnlb --
import vnlb.pylib as pyvnlb



"""


[basic]
vnlb -> step1
vnlb -> step2
pyvnlb -> step1
pyvnlb -> step2


[denoised]
vnlb -> step1 + step2
pyvnlb -> step1 + step2

[tvl1 flow]
cpp v.s. python





"""

def exec_pyvnlb(std):

    # -- denoise --
    result = pyvnlb.runPyVnlb(noisy,std)
    denoised = result['denoised']
    th_save_image(denoised,"./output/denoised.png")
    


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Please specify an integer noise level.")
        print("Examples: 10, 20, ..., 50")
        sys.exit(0)
    std = int(sys.argv[1])
    exec_pyvnlb(std)
