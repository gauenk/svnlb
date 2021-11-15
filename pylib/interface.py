
import torch
import numpy
import vnlb
from .param_parser import init_args,set_tensors,np_zero_tensors

def runPyVnlb(noisy,sigma,pyargs=None):
    if torch.is_tensor(noisy):
        noisy = noisy.cpu().numpy()
    res = runVnlb_np(noisy,sigma,pyargs)
    return res

def runVnlb_np(noisy,sigma,pyargs=None):
    
    # -- extract info --
    t,h,w,c  = noisy.shape
    args = init_args(noisy,sigma,pyargs)

    # -- create containers if needed --
    ztensors = np_zero_tensors(t,h,w,c)

    # -- set arrays --
    set_tensors(args,pyargs,ztensors)

    # -- exec using numpy --
    vnlb.runVnlb(args)

