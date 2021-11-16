import torch
import numpy as np
import vnlb

def optional(pydict,key,default,dtype=None):
    # -- get elem --
    rtn = default
    if pydict is None: rtn = default
    elif key in pydict: rtn = pydict[key]

    # -- convert to correct numpy type --
    if isinstance(rtn,list):
        if dtype is None: dtype = np.float32
        rtn = np.array(rtn,dtype=dtype)

    return rtn

def optional_swig_ptr(elem):
    if not isinstance(elem,np.ndarray):
        return elem
    # elem = np.ascontiguousarray(elem)
    return vnlb.swig_ptr(elem)

