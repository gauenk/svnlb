
# -- python imports --
import torch
import numpy
from einops import rearrange

# -- vnlb imports --
import vnlb

# -- local imports --
from .param_parser import init_args

def runPyVnlb(noisy,sigma,pyargs=None):
    if torch.is_tensor(noisy):
        noisy = noisy.cpu().numpy()
    res = runVnlb_np(noisy,sigma,pyargs)
    return res

def runVnlb_np(noisy,sigma,pyargs=None):
    
    # -- extract info --
    c,t,h,w  = noisy.shape
    args,sargs = init_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    vnlb.runVnlb(sargs)

    # -- format & create results --
    res = {}
    res['final'] = rearrange(args.final,'w h c t -> c t h w')

    # -- alias some vars --
    res['denoised'] = res['final']

    return res
