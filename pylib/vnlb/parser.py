
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

import vnlb

# from .ptr_utils import py2swig
from ..image_utils import est_sigma
from ..utils import optional,optional_swig_ptr,expand_flows,ndarray_ctg_dtype,rgb2bw

def set_optional_params(args,pyargs):
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

def check_and_expand_flows(pyargs,t):
    fflow,bflow = pyargs['fflow'],pyargs['bflow']
    nfflow = fflow.shape[0]
    nbflow = bflow.shape[0]
    assert nfflow == nbflow,"num flows must be equal."
    if nfflow == t-1:
        expand_flows(pyargs)    
    elif nfflow < t-1:
        msg = "The input flows are the wrong shape.\n"
        msg += "(nframes,two,height,width)"
        raise ValueError(msg)

def create_swig_args(args):
    sargs = vnlb.PyVnlbParams()
    for key,val in args.items():
        sval = optional_swig_ptr(val)
        setattr(sargs,key,sval)
    return sargs

def parse_args(noisy,sigma,pyargs):

    # -- extract info --
    verbose = optional(pyargs,'verbose',False)
    dtype = noisy.dtype
    t,c,h,w  = noisy.shape

    # -- format noisy image --
    noisy = ndarray_ctg_dtype(noisy,np.float32,verbose)

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
    args.sigmaBasic = np.array([0.,0.],dtype=np.float32)
    
    # -- set optional params --
    set_optional_params(args,pyargs)

    # -- format flows for c++ --
    if args.use_flow: check_and_expand_flows(pyargs,t)

    # -- create shell tensors & set arrays --
    ztensors = np_zero_tensors(t,c,h,w)
    set_tensors(args,pyargs,ztensors)

    # -- copy to swig --
    sargs = create_swig_args(args)

    return args, sargs
