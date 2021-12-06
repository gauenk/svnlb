

import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

def runFlatAreas(group,psX,psT,nSimP,chnls,gamma,sigma):
    """
    Decide if the region's area is "flat"
    """

    # -- create vars --
    pdim = psX*psX*psT
    Z = pdim*nSimP

    # -- shapes --
    gflat = group.ravel()[:pdim*nSimP*chnls]
    gflat = gflat.reshape(chnls,pdim*nSimP).T

    # -- compute var --
    gsum = np.sum(gflat,axis=0)
    gsum2 = np.sum(gflat**2,axis=0)
    var = (gsum2 - (gsum*gsum/Z)) / (Z-1)
    var = np.mean(var)

    # -- compute thresh --
    thresh = gamma*sigma**2

    # -- compare --
    flatPatch = var < thresh

    return flatPatch
