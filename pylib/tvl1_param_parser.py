"""
Parse parameters for TVL1

"""

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

def parse_args(burst,sigma,pyargs):

    # -- extract info --
    verbose = optional(pyargs,'verbose',True)
    dtype = burst.dtype
    c,t,h,w  = burst.shape

    # -- format burst image --
    burst = rearrange(burst,'c t h w -> t c h w')
    # burst = np.ascontiguousarray(np.flip(burst,axis=1).copy()) # RGB -> BGR
    if dtype != np.float32 and verbose:
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
