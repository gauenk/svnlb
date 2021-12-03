import pyvnlb
import numpy as np

def covmat_parser(groups,pdim,nSimP,chnls,rank):

    # -- create params --
    params = pyvnlb.CovMatParams()
    params.pdim = pdim
    params.rank = rank
    params.nSimP = nSimP#groups.shape[-1]
    params.gsize = groups.size

    # -- init shapes --
    covMat = np.zeros((pdim,pdim),dtype=np.float32)
    covEigVals = np.zeros((pdim),dtype=np.float32)
    covEigVecs = np.zeros((params.rank,pdim),dtype=np.float32)

    # -- tensors --
    params.groups = pyvnlb.swig_ptr(groups)
    params.covMat = pyvnlb.swig_ptr(covMat)
    params.covEigVals = pyvnlb.swig_ptr(covEigVals)
    params.covEigVecs = pyvnlb.swig_ptr(covEigVecs)

    return params,covMat,covEigVals,covEigVecs

