
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
    # -- set optional numeric vals --
    args.ps = optional(pyargs,'ps',3)
    args.k = optional(pyargs,'k',1)
    args.use_clean = not(optional(pyargs,'clean',None) is None)
    use_flow = not(type(optional(pyargs,'fflow',None)) == type(None))
    use_flow = use_flow and not(type(optional(pyargs,'bflow',None)) == type(None))
    args.use_flow = use_flow
    args.search_space = optional(pyargs,'search_space',[3,3],np.uint32)
    args.num_patches = optional(pyargs,'num_patches',[3,3],np.uint32)
    args.rank = optional(pyargs,'rank',[39,39],np.uint32)
    args.thresh = optional(pyargs,'thresh',[1e-1,1e-2])
    args.beta = optional(pyargs,'beta',[1e-1,1e-2])
    args.flat_areas = optional(pyargs,'flat_areas',[True,True],bool)
    args.couple_ch = optional(pyargs,'couple_ch',[True,True],bool)
    args.aggeBoost = optional(pyargs,'agge_boost',[True,True],bool)
    args.patch_step = optional(pyargs,'patch_step',[4,4],np.uint32)
    args.testing = optional(pyargs,'testing',False)
    args.verbose = optional(pyargs,'verbose',False)
    args.print_params = optional(pyargs,'print_params',0)
    
def np_zero_tensors(t,c,h,w):
    tensors = edict()
    tensors.fflow = np.zeros((t,2,h,w),dtype=np.float32)
    tensors.bflow = np.zeros((t,2,h,w),dtype=np.float32)
    tensors.oracle = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.clean = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.basic = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.final = np.zeros((t,c,h,w),dtype=np.float32)
    return tensors

def set_tensors(args,pyargs,tensors):
    args.fflow = optional(pyargs,'fflow',tensors.fflow)
    args.bflow = optional(pyargs,'bflow',tensors.bflow)
    args.oracle = optional(pyargs,'oracle',tensors.oracle)
    args.clean = optional(pyargs,'clean',tensors.clean)
    args.basic = optional(pyargs,'basic',tensors.basic)
    args.final = optional(pyargs,'final',tensors.final)

def create_swig_args(args):
    sargs = vnlb.PyVnlbParams()
    for key,val in args.items():
        sval = optional_swig_ptr(val)
        setattr(sargs,key,sval)
    return sargs

def expand_flows(pyargs):

    # -- unpack --
    fflow,bflow = pyargs['fflow'],pyargs['bflow']
    np.cat = np.concatenate

    # -- expand according to original c++ repo --
    fflow = np.cat([fflow,fflow[[-1]]],axis=0)
    bflow = np.cat([bflow[[0]],bflow],axis=0)

    # -- update --
    pyargs['fflow'],pyargs['bflow'] = fflow,bflow


def parse_args(noisy,sigma,pyargs):

    # -- extract info --
    verbose = optional(pyargs,'verbose',False)
    dtype = noisy.dtype
    t,c,h,w  = noisy.shape

    # -- format noisy image --
    noisy = np.ascontiguousarray(np.flip(noisy,axis=1).copy()) # RGB -> BGR
    if dtype != np.float32:
        if verbose:
            print(f"Warning: converting burst image from {dtype} to np.float32.")
        noisy = noisy.astype(np.float32)
    if not noisy.data.contiguous:
        noisy = np.ascontiguousarray(noisy)

    # -- get sigma --
    sigma = optional(pyargs,'sigma',sigma)
    if sigma is None:
        sigma = est_sigma(noisy)

    # -- params --
    args = edict()

    # -- set required numeric values --
    args.w = w
    args.h = h
    args.c = c
    args.t = t
    args.noisy = noisy
    args.sigma = np.array([sigma,sigma],dtype=np.float32)
    args.sigmaBasic = np.array([sigma,sigma],dtype=np.float32)
    
    # -- set optional params --
    set_optional_params(args,pyargs)

    # -- format flows for c++ --
    if args.use_flow: expand_flows(pyargs)

    # -- create shell tensors & set arrays --
    ztensors = np_zero_tensors(t,c,h,w)
    set_tensors(args,pyargs,ztensors)

    # -- copy to swig --
    sargs = create_swig_args(args)

    return args, sargs
