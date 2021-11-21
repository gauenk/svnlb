
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

import pyvnlb

# from .ptr_utils import py2swig
from ..image_utils import est_sigma
from ..utils import optional,optional_swig_ptr,ndarray_ctg_dtype
from ..utils import check_flows,check_none,assign_swig_args,check_and_expand_flows

#
# --Vnlb Parameters --
#

def set_function_params(args,pyargs):
    """
    args: settings for SWIG
    pyargs: settings from python
    """
    # -- set optional numeric vals --
    args.use_default = optional(pyargs,'use_default',True)

    args.ps_x = optional(pyargs,'ps_x',[-1,-1],np.int32)
    args.ps_t = optional(pyargs,'ps_t',[-1,-1],np.int32)
    args.tau = optional(pyargs,'tau',[0.,400.],np.float32)
    args.num_patches = optional(pyargs,'num_patches',[-1,-1],np.int32)
    args.sizeSearchWindow = optional(pyargs,'sizeSearchWindow',[27,27],np.int32)
    args.sizeSearchTimeFwd = optional(pyargs,'sizeSearchTimeFwd',[6,6],np.int32)
    args.sizeSearchTimeBwd = optional(pyargs,'sizeSearchTimeBwd',[6,6],np.int32)

    args.rank = optional(pyargs,'rank',[-1,-1],np.int32)
    args.thresh = optional(pyargs,'thresh',[-1.,-1.],np.float32)
    args.beta = optional(pyargs,'beta',[-1.,-1.],np.float32)

    args.flat_areas = optional(pyargs,'flat_areas',[False,True],bool)
    args.couple_ch = optional(pyargs,'couple_ch',[False,False],bool)
    args.aggreBoost = optional(pyargs,'aggre_boost',[True,True],bool)
    args.procStep = optional(pyargs,'procStep',[-1,-1],np.int32)

    args.use_clean = not(optional(pyargs,'clean',None) is None)
    use_flow = not(type(optional(pyargs,'fflow',None)) == type(None))
    use_flow = use_flow and not(type(optional(pyargs,'bflow',None)) == type(None))
    args.use_flow = use_flow

    args.testing = optional(pyargs,'testing',False)
    args.var_mode = optional(pyargs,'var_mode',False) # T == Soft, F == Hard
    args.verbose = optional(pyargs,'verbose',False)
    args.print_params = optional(pyargs,'print_params',0)

#
# -- Tensor Parser Variables --
#

def np_zero_tensors(t,c,h,w):
    tensors = edict()
    tensors.fflow = np.zeros((t,2,h,w),dtype=np.float32)
    tensors.bflow = np.zeros((t,2,h,w),dtype=np.float32)
    tensors.oracle = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.clean = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.basic = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.denoised = np.zeros((t,c,h,w),dtype=np.float32)
    return tensors

def set_tensors(args,pyargs,tensors):

    # -- set tensors --
    args.fflow = optional(pyargs,'fflow',tensors.fflow)
    args.bflow = optional(pyargs,'bflow',tensors.bflow)
    args.oracle = optional(pyargs,'oracle',tensors.oracle)
    args.clean = optional(pyargs,'clean',tensors.clean)
    args.basic = optional(pyargs,'basic',tensors.basic)
    args.denoised = optional(pyargs,'denoised',tensors.denoised)

    # -- set bools --
    args.use_flow = check_flows(pyargs)
    args.use_clean = check_none(optional(pyargs,'clean',None),'neq')
    args.use_oracle = check_none(optional(pyargs,'oracle',None),'neq')

#
# -- Main Parser --
#

def parse_args(noisy,sigma,pyargs):

    # -- extract info --
    verbose = optional(pyargs,'verbose',False)
    dtype = noisy.dtype
    t,c,h,w  = noisy.shape

    # -- format noisy image --
    noisy = ndarray_ctg_dtype(noisy,np.float32,verbose)

    # -- format flows for c++ (t-1 -> t) --
    if check_flows(pyargs): check_and_expand_flows(pyargs,t)

    # -- get sigma --
    sigma = optional(pyargs,'sigma',sigma)
    if sigma is None:
        sigma = est_sigma(noisy)

    # -- set function vars --
    args = edict()
    args.sigma = np.array([sigma,sigma],dtype=np.float32)
    args.sigmaBasic = np.array([0.,0.],dtype=np.float32)
    set_function_params(args,pyargs)

    # -- set tensors vars --
    tensors = edict()
    tensors.w = w
    tensors.h = h
    tensors.c = c
    tensors.t = t
    tensors.noisy = noisy
    ztensors = np_zero_tensors(t,c,h,w)
    set_tensors(tensors,pyargs,ztensors)

    # -- copy to swig --
    sargs = pyvnlb.PyVnlbParams()
    assign_swig_args(args,sargs)
    targs = pyvnlb.VnlbTensors()
    assign_swig_args(tensors,targs)

    return args, sargs, tensors, targs
