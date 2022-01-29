"""

Tensor parser for sim search

"""


# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

# -- package --
import svnlb

# from .ptr_utils import py2swig
from svnlb.utils.image_utils import est_sigma
from svnlb.utils.utils import optional,optional_pair,optional_swig_ptr,ndarray_ctg_dtype
from svnlb.utils.utils import check_flows,check_none,assign_swig_args,check_and_expand_flows
# from ..utils import optional,optional_swig_ptr,ndarray_ctg_dtype
# from ..utils import check_flows,check_none,assign_swig_args,check_and_expand_flows


def np_zero_tensors(t,c,h,w,groupShape,pNum,nParts):
    tensors = edict()
    tensors.fflow = np.zeros((t,2,h,w),dtype=np.float32)
    tensors.bflow = np.zeros((t,2,h,w),dtype=np.float32)
    tensors.oracle = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.clean = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.basic = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.denoised = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.groupNoisy = np.zeros(groupShape,dtype=np.float32)
    tensors.groupBasic = np.zeros(groupShape,dtype=np.float32)
    tensors.indices = np.zeros((nParts,pNum),dtype=np.uint32)
    return tensors

def set_tensors(args,pyargs,tensors):

    # -- set tensors --
    args.fflow = optional(pyargs,'fflow',tensors.fflow)
    args.bflow = optional(pyargs,'bflow',tensors.bflow)
    args.oracle = optional(pyargs,'oracle',tensors.oracle)
    args.clean = optional(pyargs,'clean',tensors.clean)
    args.basic = optional(pyargs,'basic',tensors.basic)
    args.denoised = optional(pyargs,'denoised',tensors.denoised)
    args.groupNoisy = optional(pyargs,'groupNoisy',tensors.groupNoisy)
    args.groupBasic = optional(pyargs,'groupBasic',tensors.groupBasic)
    args.indices = optional(pyargs,'indices',tensors.indices)

    # -- set bools --
    args.use_flow = check_flows(pyargs)
    args.use_clean = check_none(optional(pyargs,'clean',None),'neq')
    args.use_oracle = check_none(optional(pyargs,'oracle',None),'neq')

def combine_dicts(params_1,params_2):
    py_params = {}
    for key in params_1.keys():
        v1 = params_1[key]
        v2 = params_2[key]
        py_params[key] = [v1,v2]
    return py_params

def simSizes(params,c,nParts):
    step1 = params.isFirstStep
    sWx = params.sizeSearchWindow
    sWt = params.sizeSearchTimeFwd + params.sizeSearchTimeBwd + 1
    sPx = params.sizePatch
    sPt = params.sizePatchTime
    sPc = c if params.coupleChannels else 1
    pChn = 1 if params.coupleChannels else c
    npatches = sWx * sWx * sWt
    # [old] groupShape = (nParts,sPt,sPc,pChn,sPx,sPx,npatches)
    groupShape = (nParts,sPc*pChn,sPt,sPx,sPx,npatches)
    return groupShape,npatches

def sim_parser(noisy,sigma,nParts,py_tensors,params):

    # -- extract info --
    dtype = noisy.dtype
    t,c,h,w  = noisy.shape
    verbose = optional(params,'verbose',False)

    # -- get sim shapes --
    assert params != None, "params can not be none."
    groupShape,pNum = simSizes(params,c,nParts)

    # -- format noisy image --
    noisy = ndarray_ctg_dtype(noisy,np.float32,verbose)

    # -- format flows for c++ (t-1 -> t) --
    if check_flows(py_tensors):
        check_and_expand_flows(py_tensors,t)

    # -- set tensors vars --
    tensors = edict()
    tensors.w = w
    tensors.h = h
    tensors.c = c
    tensors.t = t
    tensors.noisy = noisy
    ztensors = np_zero_tensors(t,c,h,w,groupShape,pNum,nParts)
    set_tensors(tensors,py_tensors,ztensors)

    # -- copy to swig --
    swig_tensors = vnlb.VnlbTensors()
    assign_swig_args(tensors,swig_tensors)

    return tensors, swig_tensors

