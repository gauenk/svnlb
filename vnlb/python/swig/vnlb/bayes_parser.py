
import vnlb
import numpy
from easydict import EasyDict as edict
from vnlb.utils import assign_swig_args,get_patch_shapes_from_params

def parse_bayes_params(groupNoisy,groupBasic,nSimP,rank_var,shape,params):

    # -- init --
    bayesParams = edict()
    bayesParams.nSimP = nSimP
    bayesParams.rank_var = rank_var

    # -- allocate input vectors --
    t,c,h,w = shape
    p_num,p_dim,p_chnls = get_patch_shapes_from_params(params,c)

    # -- init groups --
    if groupNoisy is None:
        groupNoisy = numpy.zeros((p_num,p_chnls,p_dim),dtype=numpy.float32)
    if groupBasic is None:
        groupBasic = numpy.zeros((p_num,p_chnls,p_dim),dtype=numpy.float32)
    bayesParams.groupNoisy = groupNoisy
    bayesParams.groupBasic = groupBasic

    # -- allocate for vectors --
    bayesParams.mat_group = numpy.zeros((p_num,p_dim),dtype=numpy.float32)
    bayesParams.mat_center = numpy.zeros((p_chnls,p_dim),dtype=numpy.float32)
    bayesParams.mat_covMat = numpy.zeros((p_dim,p_dim),dtype=numpy.float32)
    bayesParams.mat_covEigVecs = numpy.zeros((params.rank,p_dim),dtype=numpy.float32)
    bayesParams.mat_covEigVals = numpy.zeros((p_dim),dtype=numpy.float32)

    # -- shape info --
    bayesParams.t = t
    bayesParams.c = c
    bayesParams.h = h
    bayesParams.w = w

    # -- copy to swig --
    swig_bayesParams = vnlb.PyBayesEstimateParams()
    assign_swig_args(bayesParams,swig_bayesParams)

    return bayesParams,swig_bayesParams

