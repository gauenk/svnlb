import torch
import numpy as np
import vnlb

def compute_psnrs(img1,img2,imax=255.):

    # -- same num of dims --
    assert img1.ndim == img2.ndim,"both must have same dims."

    # -- give batch dim if not exist --
    if img1.ndim == 3:
        img1 = img1[None,:]
        img2 = img2[None,:]

    # -- compute --
    eps=1e-16
    b = img1.shape[0]
    img1 = img1/255.
    img2 = img2/255.
    delta = (img1 - img2)**2
    mse = delta.reshape(b,-1).mean(axis=1) + eps
    log_mse = np.ma.log10(1./mse).filled(-np.infty)
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


def expand_flows(pydict,axis=0):
    """
    CPP requires the flows be repeated so
    the number of temporal flows matches
    the number of frames in a burst.
    """
    
    # -- unpack --
    fflow,bflow = pydict['fflow'],pydict['bflow']
    np.cat = np.concatenate

    # -- expand according to original c++ repo --
    if axis == 0:
        fflow = np.cat([fflow,fflow[[-1]]],axis=0)
        bflow = np.cat([bflow[[0]],bflow],axis=0)
    elif axis == 1:
        fflow = np.cat([fflow,fflow[:,[-1]]],axis=1)
        bflow = np.cat([bflow[:,[0]],bflow],axis=1)
    else:
        raise ValueError(f"Invalid axis {axis}")

    # -- update --
    pydict['fflow'],pydict['bflow'] = fflow,bflow


