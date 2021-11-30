
import pyvnlb
import numpy
from easydict import EasyDict as edict
from ..utils import assign_swig_args,get_patch_shapes_from_params

def parse_agg_params(deno,group,indices,weights,mask,nSimP,params):

    # -- init --
    aggParams = edict()

    # -- allocate input vectors --
    t,c,h,w = deno.shape
    shape = deno.shape
    p_num,p_dim,p_chnls = get_patch_shapes_from_params(params,c)

    # -- [updated] denoised image --
    if deno is None:
        deno = numpy.zeros(shape,dtype=numpy.float32)
    aggParams.imDeno = deno

    # -- [updated] weights used for division later --
    if weights is None:
        weights = numpy.zeros((t,1,h,w),dtype=numpy.float32)
    aggParams.weights = weights

    # -- [updated] mask --
    if mask is None:
        mask = numpy.zeros((t,1,h,w),dtype=numpy.int8)
    aggParams.mask = mask

    # -- [updated] num of "complete" pixels --
    aggParams.nmasked = 0

    # -- group used for denoising --
    aggParams.group = group
    groupSize = p_num * p_dim * p_chnls
    assert group.size == groupSize

    # -- indices used for placement in Denoised image --
    aggParams.indices = indices

    # -- shape info --
    aggParams.t = t
    aggParams.c = c
    aggParams.h = h
    aggParams.w = w

    # -- num of similar patches --
    aggParams.nSimP = nSimP

    # -- copy to swig --
    swig_aggParams = pyvnlb.PyAggParams()
    assign_swig_args(aggParams,swig_aggParams)
    swig_aggParams.mask = pyvnlb.swig_ptr(aggParams.mask)

    return aggParams,swig_aggParams

