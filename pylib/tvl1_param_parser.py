"""
Parse parameters for TVL1

"""

import cv2
import torch
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

import vnlb

# from .ptr_utils import py2swig
from .image_utils import est_sigma
from .utils import optional,optional_swig_ptr

def set_optional_params(args,pyargs):

    args.nproc = optional(pyargs,'nproc',-1)
    args.tau = optional(pyargs,'tau',-1)
    args.plambda = optional(pyargs,'lambda',-1) 
    args.theta = optional(pyargs,'theta',-1)  
    args.nscales = optional(pyargs,'nscales',-1)
    args.fscale = optional(pyargs,'fscale',-1) 
    args.zfactor = optional(pyargs,'zfactor',-1)
    args.nwarps = optional(pyargs,'nwarps',-1) 
    args.epsilon = optional(pyargs,'epsilon',-1)
    args.verbose = optional(pyargs,'verbose',False)
    args.testing = optional(pyargs,'testing',False)
    args.direction = optional(pyargs,'direction',0)
    
def np_zero_tensors(t,c,h,w):
    tensors = edict()
    tensors.fflow = np.zeros((t-1,2,h,w),dtype=np.float32)
    tensors.bflow = np.zeros((t-1,2,h,w),dtype=np.float32)
    return tensors

def set_tensors(args,pyargs,tensors):
    args.fflow = optional(pyargs,'fflow',tensors.fflow)
    args.bflow = optional(pyargs,'bflow',tensors.bflow)

def create_swig_args(args):
    sargs = vnlb.PyTvFlowParams()
    for key,val in args.items():
        sval = optional_swig_ptr(val)
        setattr(sargs,key,sval)
    return sargs

def rgb2bw(burst):
    print(burst.shape)
    # burst_bw = .299 * burst[:,2] + .587 * burst[:,1] + .114 * burst[:,0]
    # burst_bw = burst_bw[:,None].copy()
    burst = burst.copy()
    burst_bw = []
    for t in range(burst.shape[0]):
        frame = burst[t]
        frame = np.ascontiguousarray(rearrange(frame,'c h w -> h w c')).copy()
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        frame = rearrange(frame,'h w -> 1 h w')
        burst_bw.append(frame)
    burst_bw = np.stack(burst_bw).copy()
    # import torch
    # import torchvision.utils as tvUtils
    # tvUtils.save_image(torch.FloatTensor(burst_bw)/255.,"burst_bw_inner.png")
    
    
    return burst_bw

def parse_args(burst,sigma,pyargs):

    # -- extract info --
    dtype = burst.dtype
    use_rgb2bw = optional(pyargs,'bw',False)
    verbose = optional(pyargs,'verbose',False)
    if use_rgb2bw: burst = rgb2bw(burst)
    t,c,h,w  = burst.shape

    # -- format burst image --
    burst = np.ascontiguousarray(burst)
    if dtype != np.float32:
        if verbose:
            print(f"Warning: converting burst image from {dtype} to np.float32.")
        burst = burst.astype(np.float32)
    if not burst.data.contiguous:
        burst = np.ascontiguousarray(burst)

    # -- get sigma --
    sigma = optional(pyargs,'sigma',sigma)
    if sigma is None:
        sigma = est_sigma(burst)

    # -- params --
    args = edict()

    # -- set required numeric values --
    args.w = w
    args.h = h
    args.c = c
    args.t = t
    args.burst = burst
    
    # -- set optional params --
    set_optional_params(args,pyargs)

    # -- create shell tensors & set arrays --
    ztensors = np_zero_tensors(t,c,h,w)
    set_tensors(args,pyargs,ztensors)

    # -- copy to swig --
    sargs = create_swig_args(args)

    return args, sargs
