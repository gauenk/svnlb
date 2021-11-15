
# -- python imports --
import torch
import numpy
from einops import rearrange

# -- vnlb imports --
import vnlb

# -- local imports --
from .param_parser import parse_args

def runPyVnlb(noisy,sigma,pyargs=None):
    if torch.is_tensor(noisy):
        noisy = noisy.cpu().numpy()
    res = runVnlb_np(noisy,sigma,pyargs)
    return res

def runVnlb_np(noisy,sigma,pyargs=None):
    
    # -- extract info --
    c,t,h,w  = noisy.shape
    args,sargs = parse_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    vnlb.runVnlb(sargs)

    # -- format & create results --
    res = {}
    res['final'] = rearrange(args.final,'t c h w -> c t h w')

    # -- alias some vars --
    res['denoised'] = res['final']

    return res
