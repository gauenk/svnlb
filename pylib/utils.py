import torch
import numpy as np
import vnlb

def np_log(np_array):
    if type(np_array) is not np.ndarray:
        if type(np_array) is not list:
            np_array = [np_array]
        np_array = np.array(np_array)
    return np.ma.log(np_array).filled(-np.infty)

def psnrs(img1,img2,imax=255.):
    eps=1e-16
    img1 = img1/255.
    img2 = img2/255.
    mse = ((img1-img2)**2).mean() + eps
    log_mse = np_log10(1./mse).filled(-np.inty)
    psnr = 10 * log_mse
    return psnr

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

