"""

Tensor parser for sim search

"""


# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

# -- package --
import pyvnlb

# from .ptr_utils import py2swig
from ..image_utils import est_sigma
from ..utils import optional,optional_swig_ptr,ndarray_ctg_dtype
from ..utils import check_flows,check_none,assign_swig_args,check_and_expand_flows


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
    # pDim = sPx * sPx * sPt * (params.coupleChannels ? c : 1)
    npatches = sWx * sWx * sWt
    # return pNum,pDim,pChn
    # groupShape = (nParts,sPt,sPc,pChn,sPx,sPx,npatches)
    groupShape = (nParts,sPt,sPc*pChn,sPx,sPx,npatches)
    # groupShape = (pNum,sPx,sPx,sPt,pChn,sPc)
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
    if check_flows(py_tensors): check_and_expand_flows(py_tensors,t)

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
    swig_tensors = pyvnlb.VnlbTensors()
    assign_swig_args(tensors,swig_tensors)

    return tensors, swig_tensors

# def reorder_sim_group(group,psX,psT,c,nSimP):
#     """
#     The patch data is not contiguous and this code
#     corrects this through reshapes (creating new strides)
#     and concatenations (pasting two edges together).

#     E.x.
#     Idx to Access inside a Match: 0,...,100,..,200,..,(psX**2*psT*c)*100
#     Idx to Access the Patch (px**2): [0],[2],[4],[1],[3],[5]...
#     Idx to Access the Time/Color Channels of the Image...
#     """
#     ncat = np.concatenate
#     numNz = nSimP * psX * psX * psT * c
#     group_f = group.ravel()[:numNz]
#     group = group_f.reshape(c,psT,-1)
#     group = ncat(group,axis=1)
#     group_f = group.ravel()
#     group = group_f.reshape(c*psT,psX**2,nSimP).transpose(2,0,1)
#     group = ncat(group,axis=1)
#     group = group.reshape(c*psT,nSimP,psX**2).transpose(1,0,2)
#     group = ncat(group,axis=0)
#     group = group.reshape(nSimP,psT,c,psX,psX)
#     return group

