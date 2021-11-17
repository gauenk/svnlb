import torch
import torchvision.utils as tvUtils

import numpy as np
from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

def th_save_image(burst,fn,imax=1.):
    burst = torch.FloatTensor(burst.copy())/imax
    # if burst.shape[0] in [1,3,4]:
    #     burst = rearrange(burst,'c t h w -> t c h w')
    tvUtils.save_image(burst,fn)


def save_field(field,cppField,pyField):
    # print("cppField.shape: ",cppField.shape)
    # print("pyField.shape: ",pyField.shape)
    if field in ["fflow","bflow"]:
        th_save_image(cppField[[0]],f"cpp_{field}_0.png")
        th_save_image(cppField[[1]],f"cpp_{field}_1.png")
        
        th_save_image(pyField[[0]],f"py_{field}_0.png")
        th_save_image(pyField[[1]],f"py_{field}_1.png")
    elif field in ["denoised","basic"]:
        th_save_image(cppField/255.,f"cpp_{field}.png")
        th_save_image(pyField/255.,f"py_{field}.png")
        delta = cppField - pyField
        # print(delta.min(),delta.max(),delta.mean())
        # delta = delta + delta.min()
        # delta /= delta.max()
        th_save_image(delta,f"delta_{field}.png")
    else:
        print(f"Uknown save for field [{field}]")

def relative_error(approx,gt):
    eps = 1e-16
    rel = np.abs(approx-gt)/(gt+eps)
    return np.mean(rel)
    
def np_log(np_array):
    if type(np_array) is not np.ndarray:
        if type(np_array) is not list:
            np_array = [np_array]
        np_array = np.array(np_array)
    return np.ma.log(np_array).filled(-np.infty)
        
def compute_psnrs(img1,img2):
    eps=1e-16
    img1 = img1/255.
    img2 = img2/255.
    mse = ((img1-img2)**2).mean() + eps
    psnr = 10 * np_log(1./mse)[0]/np_log(10)[0]
    return psnr

