import vnlb
import numpy as np

def covmat_parser(groups,pdim,nSimP,chnls,rank):

    # -- create params --
    params = vnlb.CovMatParams()
    params.pdim = pdim
    params.rank = rank
    params.nSimP = nSimP#groups.shape[-1]
    params.gsize = groups.size

    # -- init shapes --
    covMat = np.zeros((pdim,pdim),dtype=np.float32)
    covEigVals = np.zeros((pdim),dtype=np.float32)
    covEigVecs = np.zeros((params.rank,pdim),dtype=np.float32)

    # -- tensors --
    params.groups = vnlb.swig_ptr(groups)
    params.covMat = vnlb.swig_ptr(covMat)
    params.covEigVals = vnlb.swig_ptr(covEigVals)
    params.covEigVecs = vnlb.swig_ptr(covEigVecs)

    return params,covMat,covEigVals,covEigVecs

