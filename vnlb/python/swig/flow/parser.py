"""
Parse parameters for TVL1

"""

import cv2
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

import vnlb

# from .ptr_utils import py2swig
from vnlb.utils import est_sigma
from vnlb.utils import optional,optional_swig_ptr,ndarray_ctg_dtype,rgb2bw
from vnlb.utils import check_none,assign_swig_args


#
# -- Flow Function --
#

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


#
# -- Tensors
#

def np_zero_tensors(t,c,h,w):
    tensors = edict()
    tensors.fflow = np.zeros((t-1,2,h,w),dtype=np.float32)
    tensors.bflow = np.zeros((t-1,2,h,w),dtype=np.float32)
    tensors.clean = np.zeros((t,c,h,w),dtype=np.float32)
    return tensors

def set_tensors(targs,pyargs,tensors):
    targs.fflow = optional(pyargs,'fflow',tensors.fflow)
    targs.bflow = optional(pyargs,'bflow',tensors.bflow)
    targs.clean = optional(pyargs,'clean',tensors.clean)

    targs.use_clean = check_none(optional(pyargs,'clean',None),'neq')
    targs.use_flow = True
    targs.use_oracle = False

#
# -- Main --
#

def parse_args(burst,sigma,pyargs):

    # -- extract info --
    dtype = burst.dtype
    use_rgb2bw = optional(pyargs,'bw',True)
    verbose = optional(pyargs,'verbose',False)
    if use_rgb2bw: burst = rgb2bw(burst)
    t,c,h,w  = burst.shape

    # -- format burst image --
    if c == 1: burst = burst[:,0]
    burst = ndarray_ctg_dtype(burst,np.float32,verbose)

    # -- get sigma --
    sigma = optional(pyargs,'sigma',sigma)
    if sigma is None:
        sigma = est_sigma(burst)

    # -- set function params --
    args = edict()
    set_optional_params(args,pyargs)

    # -- set tensor params --
    tensors = edict()
    tensors.w = w
    tensors.h = h
    tensors.c = c
    tensors.t = t
    tensors.noisy = burst
    ztensors = np_zero_tensors(t,c,h,w)
    set_tensors(tensors,pyargs,ztensors)

    # -- copy to swig --
    sargs = vnlb.PyTvFlowParams()
    assign_swig_args(args,sargs)

    # -- copy to swig --
    targs = vnlb.VnlbTensors()
    assign_swig_args(tensors,targs)

    return args, sargs, tensors, targs
